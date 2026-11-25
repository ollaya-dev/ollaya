//! Exact state forking for decoder graphs in the fork form (`contract.form` `"fork"`): a request's
//! questions share the state at the start of their rows, so the shared prefix runs once and each
//! question's remainder (its branch) continues it, instead of every row repeating the state.
//!
//! A fork graph takes `prefix_ids` [J, P], `branch_ids` [B, S] and `branch_job` [B]: branch b is the
//! row `prefix_ids[branch_job[b]] ++ branch_ids[b]`, and every position a graph reads counts from the
//! start of that row. P and S are multiples of the graph's chunk (64), so the Gated DeltaNet chunks
//! are the row form's and the numbers are the row form's (`convert/ollaya_convert/families/llm_common/qwen35.py`,
//! `Qwen35Trunk.fork`; [`split`] mirrors its `fork_rows`).

use std::borrow::Cow;
use std::ops::Range;
use std::sync::Mutex;

use ndarray::{Array1, Array2, Ix2};
use ort::session::{Session, SessionInputValue};
use ort::value::Tensor;

use crate::Error;
use crate::onnx::Device;

/// A graph input by name, as `session.run` takes it.
pub type Input = (Cow<'static, str>, SessionInputValue<'static>);

/// Whether the session's graph is in the fork form (it takes `prefix_ids`).
pub fn is_fork(session: &Mutex<Session>) -> bool {
    session
        .lock()
        .expect("session mutex poisoned")
        .inputs()
        .iter()
        .any(|i| i.name() == "prefix_ids")
}

/// Each request row's `output` (a graph output [rows, width]) from a fork graph, in row order, over
/// `runs` (from [`runs`] with the same `budget`). `reads` gives a run's other inputs for the
/// request rows it answers: its read positions, which count from the start of each row.
pub fn outputs(
    session: &Mutex<Session>,
    runs: Vec<Run>,
    budget: usize,
    device: Device,
    output: &str,
    reads: impl Fn(&[usize]) -> Result<Vec<Input>, Error>,
) -> Result<Vec<Vec<f32>>, Error> {
    each(runs, |run| {
        let options = crate::engine::run_options(device, run.tokens(), budget)?;
        let mut inputs: Vec<Input> = vec![
            (
                "prefix_ids".into(),
                Tensor::from_array(run.prefix_ids)?.into(),
            ),
            (
                "branch_ids".into(),
                Tensor::from_array(run.branch_ids)?.into(),
            ),
            (
                "branch_job".into(),
                Tensor::from_array(run.branch_job)?.into(),
            ),
        ];
        inputs.extend(reads(&run.rows)?);
        let mut session = session.lock().expect("session mutex poisoned");
        let result = session.run_with_options(inputs, &options)?;
        Ok(result
            .get(output)
            .ok_or_else(|| Error::Model(format!("the graph has no output {output}")))?
            .try_extract_array::<f32>()?
            .into_dimensionality::<Ix2>()
            .map_err(|e| Error::Model(format!("{output}: {e}")))?
            .to_owned())
    })
}

/// Each request row's output over `runs`, in row order: `f` answers one run with an output [its
/// rows, width].
pub fn each(
    runs: Vec<Run>,
    mut f: impl FnMut(Run) -> Result<Array2<f32>, Error>,
) -> Result<Vec<Vec<f32>>, Error> {
    let mut out = vec![Vec::new(); runs.iter().map(|r| r.rows.len()).sum()];
    for run in runs {
        let picked = run.rows.clone();
        let y = f(run)?;
        if y.nrows() != picked.len() {
            return Err(Error::Model(format!(
                "a fork run gave {} rows for {} questions",
                y.nrows(),
                picked.len()
            )));
        }
        for (&i, row) in picked.iter().zip(y.rows()) {
            out[i] = row.to_vec();
        }
    }
    Ok(out)
}

/// One `session.run` of a fork graph: the request rows it answers, in branch order, and its inputs.
#[derive(Debug, Clone, PartialEq)]
pub struct Run {
    pub rows: Vec<usize>,
    pub prefix_ids: Array2<i64>,
    pub branch_ids: Array2<i64>,
    pub branch_job: Array1<i64>,
}

impl Run {
    /// Tokens the run computes (padding included).
    pub fn tokens(&self) -> usize {
        self.prefix_ids.len() + self.branch_ids.len()
    }
}

fn up(n: usize, multiple: usize) -> usize {
    n.div_ceil(multiple) * multiple
}

fn shared(rows: &[&[u32]]) -> usize {
    let shortest = rows.iter().map(|r| r.len()).min().unwrap_or(0);
    (0..shortest)
        .find(|&i| rows.iter().any(|r| r[i] != rows[0][i]))
        .unwrap_or(shortest)
}

/// Split `rows` (every row non-empty) into one run's inputs. Rows that share `multiple` tokens or more
/// fork at their longest common prefix, rounded down to `multiple` and short of every row's end (one
/// job). Otherwise each row is its own job, cut one chunk before the end of the longest row (a
/// shorter row sits whole in its prefix, padded, and its branch is padding nothing reads), which
/// costs what the row form costs; rows of one chunk or less run in a one-chunk prefix.
pub fn split(rows: &[&[u32]], multiple: usize, pad: u32) -> Run {
    let shortest = rows.iter().map(|r| r.len()).min().unwrap_or(0);
    let longest = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    let fork = shared(rows).min(shortest.saturating_sub(1)) / multiple * multiple;
    let (p, jobs): (usize, Vec<i64>) = if fork > 0 {
        (fork, vec![0; rows.len()])
    } else {
        let p = up(longest, multiple).saturating_sub(multiple).max(multiple);
        (p, (0..rows.len() as i64).collect())
    };
    let n_prefix = if fork > 0 { 1 } else { rows.len() };
    let s = up(
        rows.iter()
            .map(|r| r.len().saturating_sub(p))
            .max()
            .unwrap_or(0),
        multiple,
    )
    .max(multiple);
    let mut prefix_ids = Array2::from_elem((n_prefix, p), i64::from(pad));
    let mut branch_ids = Array2::from_elem((rows.len(), s), i64::from(pad));
    for (j, row) in rows.iter().take(n_prefix).enumerate() {
        for (c, &id) in row.iter().take(p).enumerate() {
            prefix_ids[[j, c]] = i64::from(id);
        }
    }
    for (b, row) in rows.iter().enumerate() {
        for (c, &id) in row.iter().skip(p).enumerate() {
            branch_ids[[b, c]] = i64::from(id);
        }
    }
    Run {
        rows: (0..rows.len()).collect(),
        prefix_ids,
        branch_ids,
        branch_job: Array1::from(jobs),
    }
}

/// The runs that answer `rows` within `budget` padded tokens each (a single row over it runs alone)
/// and at most `max_rows` branches. When the rows share a prefix, each run carries it once and as
/// many branches as fit beside it (at least half the budget's worth); otherwise rows are grouped by
/// length as the row form groups them, and each group is split on its own.
pub fn runs(
    rows: &[&[u32]],
    multiple: usize,
    pad: u32,
    budget: usize,
    max_rows: usize,
) -> Vec<Run> {
    if rows.is_empty() {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..rows.len()).collect();
    order.sort_by_key(|&i| rows[i].len());
    let shortest = rows[order[0]].len();
    let p = shared(rows).min(shortest.saturating_sub(1)) / multiple * multiple;
    // With a shared prefix, a run is the prefix once plus its branches. A prefix near or over the
    // budget still leaves the branches half of it: a run then peaks at about one long row, as a
    // row over the budget does in the row form, instead of every branch recomputing the prefix.
    let (lens, room): (Vec<usize>, usize) = if p > 0 {
        let lens = order
            .iter()
            .map(|&i| up(rows[i].len() - p, multiple))
            .collect();
        (lens, budget.saturating_sub(p).max(budget / 2))
    } else {
        (
            order.iter().map(|&i| up(rows[i].len(), multiple)).collect(),
            budget,
        )
    };
    crate::engine::batches(&lens, room, max_rows)
        .into_iter()
        .map(|range: Range<usize>| {
            let picked: Vec<usize> = order[range].to_vec();
            let group: Vec<&[u32]> = picked.iter().map(|&i| rows[i]).collect();
            let mut run = if p > 0 {
                // every group forks at the request's prefix, so all of them share one length
                split_at(&group, p, multiple, pad)
            } else {
                split(&group, multiple, pad)
            };
            run.rows = picked;
            run
        })
        .collect()
}

/// `split` with the fork point given: one job, prefix `rows[0][..p]`.
fn split_at(rows: &[&[u32]], p: usize, multiple: usize, pad: u32) -> Run {
    let s = up(
        rows.iter().map(|r| r.len() - p).max().unwrap_or(0),
        multiple,
    );
    let mut prefix_ids = Array2::from_elem((1, p), i64::from(pad));
    for (c, &id) in rows[0][..p].iter().enumerate() {
        prefix_ids[[0, c]] = i64::from(id);
    }
    let mut branch_ids = Array2::from_elem((rows.len(), s), i64::from(pad));
    for (b, row) in rows.iter().enumerate() {
        for (c, &id) in row[p..].iter().enumerate() {
            branch_ids[[b, c]] = i64::from(id);
        }
    }
    Run {
        rows: (0..rows.len()).collect(),
        prefix_ids,
        branch_ids,
        branch_job: Array1::zeros(rows.len()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PAD: u32 = 9;

    fn row(shared: usize, own: usize, tag: u32) -> Vec<u32> {
        (0..shared as u32).chain((0..own).map(|_| tag)).collect()
    }

    /// The row a branch stands for: its prefix's tokens, then its own, without padding.
    fn rebuilt(run: &Run, b: usize, len: usize) -> Vec<u32> {
        let j = run.branch_job[b] as usize;
        run.prefix_ids
            .row(j)
            .iter()
            .chain(run.branch_ids.row(b).iter())
            .take(len)
            .map(|&x| x as u32)
            .collect()
    }

    #[test]
    fn rows_that_share_a_long_state_fork_at_a_chunk_boundary() {
        let rows = [row(150, 30, 1), row(150, 90, 2), row(150, 5, 3)];
        let refs: Vec<&[u32]> = rows.iter().map(|r| r.as_slice()).collect();
        let run = split(&refs, 64, PAD);
        assert_eq!(run.prefix_ids.dim(), (1, 128));
        assert_eq!(run.branch_ids.dim(), (3, 128)); // 22 + 90 tokens, up to 128
        assert_eq!(run.branch_job.to_vec(), vec![0, 0, 0]);
        for (b, r) in rows.iter().enumerate() {
            assert_eq!(rebuilt(&run, b, r.len()), *r);
        }
    }

    #[test]
    fn rows_without_a_shared_chunk_cost_what_the_row_form_costs() {
        let rows = [row(10, 120, 1), row(10, 200, 2), row(5, 20, 3)];
        let refs: Vec<&[u32]> = rows.iter().map(|r| r.as_slice()).collect();
        let run = split(&refs, 64, PAD);
        // longest 210 tokens -> 256 padded: prefixes cut at 192, one chunk of branch
        assert_eq!(run.prefix_ids.dim(), (3, 192));
        assert_eq!(run.branch_ids.dim(), (3, 64));
        assert_eq!(run.tokens(), 3 * 256);
        assert_eq!(run.branch_job.to_vec(), vec![0, 1, 2]);
        for (b, r) in rows.iter().enumerate() {
            assert_eq!(rebuilt(&run, b, r.len()), *r);
        }
    }

    #[test]
    fn rows_of_one_chunk_run_whole_in_the_prefix() {
        let rows = [row(5, 20, 1), row(5, 50, 2)];
        let refs: Vec<&[u32]> = rows.iter().map(|r| r.as_slice()).collect();
        let run = split(&refs, 64, PAD);
        assert_eq!(run.prefix_ids.dim(), (2, 64));
        assert_eq!(run.branch_ids.dim(), (2, 64));
        assert!(run.branch_ids.iter().all(|&x| x == i64::from(PAD)));
        for (b, r) in rows.iter().enumerate() {
            assert_eq!(rebuilt(&run, b, r.len()), *r);
        }
    }

    #[test]
    fn a_branch_never_comes_out_empty() {
        // identical rows of exactly two chunks: the fork stops a chunk short of their end
        let rows = [row(128, 0, 0), row(128, 0, 0)];
        let refs: Vec<&[u32]> = rows.iter().map(|r| r.as_slice()).collect();
        let run = split(&refs, 64, PAD);
        assert_eq!(run.prefix_ids.dim(), (1, 64));
        assert_eq!(run.branch_ids.dim(), (2, 64));
    }

    #[test]
    fn runs_carry_the_prefix_once_and_split_branches_by_budget() {
        let rows: Vec<Vec<u32>> = (0..6).map(|t| row(300, 100, t)).collect();
        let refs: Vec<&[u32]> = rows.iter().map(|r| r.as_slice()).collect();
        // prefix 256; each branch 144 tokens -> 192 padded; budget 1000 leaves 744: three per run
        let runs = runs(&refs, 64, PAD, 1000, usize::MAX);
        assert_eq!(runs.len(), 2);
        let mut seen: Vec<usize> = runs.iter().flat_map(|r| r.rows.clone()).collect();
        seen.sort();
        assert_eq!(seen, (0..6).collect::<Vec<_>>());
        for run in &runs {
            assert_eq!(run.prefix_ids.dim(), (1, 256));
            assert!(run.tokens() <= 1000);
            for (b, &i) in run.rows.iter().enumerate() {
                assert_eq!(rebuilt(run, b, rows[i].len()), rows[i]);
            }
        }
    }

    #[test]
    fn a_prefix_over_the_budget_still_takes_many_branches_per_run() {
        // nimble: every row carries the whole request, only the requested field differs
        let rows: Vec<Vec<u32>> = (0..40).map(|t| row(3008, 20, t)).collect();
        let refs: Vec<&[u32]> = rows.iter().map(|r| r.as_slice()).collect();
        let runs = runs(&refs, 64, PAD, 2048, usize::MAX);
        // branches of 64 padded tokens in half the budget: 16 per run
        assert_eq!(runs.len(), 3);
        assert!(runs.iter().all(|r| r.prefix_ids.dim() == (1, 3008)));
        assert_eq!(runs.iter().map(|r| r.rows.len()).sum::<usize>(), 40);
    }

    #[test]
    fn runs_without_a_shared_prefix_group_rows_like_the_row_form() {
        let rows = [row(3, 500, 1), row(3, 60, 2), row(3, 700, 3)];
        let refs: Vec<&[u32]> = rows.iter().map(|r| r.as_slice()).collect();
        let runs = runs(&refs, 64, PAD, 1024, usize::MAX);
        let mut seen: Vec<usize> = runs.iter().flat_map(|r| r.rows.clone()).collect();
        seen.sort();
        assert_eq!(seen, vec![0, 1, 2]);
        for run in &runs {
            for (b, &i) in run.rows.iter().enumerate() {
                assert_eq!(rebuilt(run, b, rows[i].len()), rows[i]);
            }
        }
    }
}
