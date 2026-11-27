import { btnPrimary, PageHead, textLink } from '../components/ui'
import { asset } from '../lib/assets'
import { CLOUD } from '../site'

// The pages are static. public/static/waitlist.js sends the form to Ollaya Cloud's sign-in service on this host,
// with the Cloudflare Turnstile token against spam. The service checks the email and keeps one row for each email
// and kind; people get access by hand during the beta.

const label = 'block text-[13px] font-medium text-muted'
const field =
  'mt-2 block w-full rounded-md border border-line bg-canvas px-3.5 py-2.5 text-[15px] text-fg hover:border-line-strong focus:border-line-strong focus:ring-4 focus:ring-fill-strong focus:outline-none'

type Kind = 'cloud' | 'inference'

function Field({ id, name, text, type = 'text', required = false, autocomplete }: { id: string; name: string; text: string; type?: string; required?: boolean; autocomplete?: string }) {
  return (
    <div>
      <label for={id} class={label}>
        {text}
        {required ? null : <span class="font-normal"> (optional)</span>}
      </label>
      <input id={id} name={name} type={type} required={required} autocomplete={autocomplete} maxlength={type === 'email' ? 254 : 120} class={field} />
    </div>
  )
}

function WaitlistForm({ kind }: { kind: Kind }) {
  return (
    <form class="max-w-xl space-y-5" data-waitlist={kind} data-endpoint={CLOUD.waitlistApi} novalidate>
      <Field id="waitlist-email" name="email" text="Email" type="email" required autocomplete="email" />
      <Field id="waitlist-name" name="name" text="Name" autocomplete="name" />
      <Field id="waitlist-company" name="company" text="Company" autocomplete="organization" />
      {kind === 'inference' ? (
        <>
          <div>
            <label for="waitlist-use-case" class={label}>
              What do you want to run?
            </label>
            <textarea id="waitlist-use-case" name="useCase" rows={4} required maxlength={2000} class={`${field} resize-y leading-relaxed`}></textarea>
          </div>
          <div>
            <label for="waitlist-volume" class={label}>
              Expected volume<span class="font-normal"> (optional)</span>
            </label>
            <select id="waitlist-volume" name="volume" class={field}>
              <option value="">Not sure yet</option>
              <option value="under-10k">Under 10,000 requests a day</option>
              <option value="10k-1m">10,000 to 1 million requests a day</option>
              <option value="over-1m">Over 1 million requests a day</option>
            </select>
          </div>
        </>
      ) : null}
      <div class="cf-turnstile" data-sitekey={CLOUD.turnstileSiteKey} data-theme="auto" data-size="flexible"></div>
      <div class="flex flex-wrap items-center gap-x-4 gap-y-3">
        <button type="submit" class={`${btnPrimary} disabled:opacity-50`} data-submit>
          {kind === 'inference' ? 'Apply for inference access' : 'Join the waitlist'}
        </button>
        <p class="text-sm text-body" role="status" aria-live="polite" data-status></p>
      </div>
    </form>
  )
}

function WaitlistScripts() {
  return (
    <>
      <script src="https://challenges.cloudflare.com/turnstile/v0/api.js" async defer></script>
      <script src={asset('waitlist.js')} defer></script>
    </>
  )
}

export function WaitlistPage() {
  return (
    <>
      <PageHead
        kicker="Ollaya Cloud beta"
        title="Join the Ollaya Cloud waitlist"
        lead="Ollaya Cloud runs Ollaya's decision models for you, behind one API and a dashboard. During the beta we add people by hand, so leave your email and we write to you when your access is ready."
      />
      <section class="rule px-6 py-10 md:px-10 md:py-12">
        <WaitlistForm kind="cloud" />
        <p class="mt-8 max-w-xl text-sm text-body">
          Do you want to call the models from your own code?{' '}
          <a href="/waitlist/inference" class={textLink}>
            Apply for inference access
          </a>
          . Already on the list?{' '}
          <a href="/dashboard/sign-in" class={textLink}>
            Sign in
          </a>
          .
        </p>
      </section>
      <WaitlistScripts />
    </>
  )
}

export function InferenceWaitlistPage() {
  return (
    <>
      <PageHead
        kicker="Ollaya Cloud beta"
        title="Apply for inference access"
        lead="Call Ollaya's decision models from your code without running them yourself. Tell us what you want to run, and we write to you when your API access is ready."
      />
      <section class="rule px-6 py-10 md:px-10 md:py-12">
        <WaitlistForm kind="inference" />
        <p class="mt-8 max-w-xl text-sm text-body">
          You can also run every model on your own machine today:{' '}
          <a href="/download" class={textLink}>
            download Ollaya
          </a>
          .
        </p>
      </section>
      <WaitlistScripts />
    </>
  )
}
