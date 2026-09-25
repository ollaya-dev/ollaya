//! [`Worker`]: the one thread that talks to MLX in a runner process.
//!
//! MLX builds graphs and encodes Metal work on the calling thread, and its streams are not meant
//! to be shared between threads that build graphs at the same time. So a runner owns exactly one
//! MLX thread: the model's arrays live there, and requests reach it as jobs over a channel, one at
//! a time. Ollama's MLX runner does the same (`mlx/mlxthread`).

use std::sync::Mutex;
use std::sync::mpsc;
use std::thread::JoinHandle;

use crate::{Error, MetalInfo, Result, metal};

type Job<M> = Box<dyn FnOnce(&M) + Send>;

/// A thread that owns a model of type `M` (typically holding [`crate::Array`]s, which never leave
/// it) and runs jobs against it.
pub struct Worker<M> {
    jobs: Mutex<Option<mpsc::Sender<Job<M>>>>,
    thread: Option<JoinHandle<()>>,
    info: MetalInfo,
}

impl<M: 'static> Worker<M> {
    /// Start the MLX thread: check the environment, point MLX at `metallib`, run the self-check,
    /// then build the model with `init` on that thread.
    pub fn start<F>(name: &str, metallib: &std::path::Path, init: F) -> Result<Self>
    where
        F: FnOnce() -> Result<M> + Send + 'static,
    {
        let metallib = metallib.to_path_buf();
        let (ready_tx, ready_rx) = mpsc::sync_channel::<Result<MetalInfo>>(1);
        let (jobs_tx, jobs_rx) = mpsc::channel::<Job<M>>();
        let thread = std::thread::Builder::new()
            .name(name.to_owned())
            .spawn(move || {
                let model = metal::start(&metallib).and_then(|info| Ok((info, init()?)));
                let model = match model {
                    Ok((info, model)) => {
                        let _ = ready_tx.send(Ok(info));
                        model
                    }
                    Err(e) => {
                        let _ = ready_tx.send(Err(e));
                        crate::unbind_stream();
                        return;
                    }
                };
                for job in jobs_rx {
                    job(&model);
                }
                // Free the model's arrays on this thread, before its stream goes.
                drop(model);
                crate::unbind_stream();
            })
            .map_err(|e| Error(format!("spawn the MLX thread: {e}")))?;
        let info = match ready_rx.recv() {
            Ok(Ok(info)) => info,
            Ok(Err(e)) => {
                let _ = thread.join();
                return Err(e);
            }
            Err(_) => {
                let _ = thread.join();
                return Err(Error("the MLX thread stopped while loading".into()));
            }
        };
        Ok(Worker {
            jobs: Mutex::new(Some(jobs_tx)),
            thread: Some(thread),
            info,
        })
    }

    /// What the self-check found.
    pub fn info(&self) -> &MetalInfo {
        &self.info
    }

    /// Run `job` on the MLX thread and wait for its result.
    pub fn run<R, F>(&self, job: F) -> Result<R>
    where
        R: Send + 'static,
        F: FnOnce(&M) -> Result<R> + Send + 'static,
    {
        let (tx, rx) = mpsc::sync_channel(1);
        let job: Job<M> = Box::new(move |model| {
            let _ = tx.send(job(model));
        });
        {
            let jobs = self.jobs.lock().map_err(|_| stopped())?;
            jobs.as_ref()
                .ok_or_else(stopped)?
                .send(job)
                .map_err(|_| stopped())?;
        }
        rx.recv().map_err(|_| stopped())?
    }
}

fn stopped() -> Error {
    Error("the MLX thread has stopped".into())
}

impl<M> Drop for Worker<M> {
    fn drop(&mut self) {
        // Closing the channel ends the thread's loop.
        if let Ok(mut jobs) = self.jobs.lock() {
            jobs.take();
        }
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
