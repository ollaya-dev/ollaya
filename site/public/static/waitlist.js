// The waitlist forms of /waitlist and /waitlist/inference (src/pages/waitlist.tsx). The form goes to Ollaya Cloud's
// sign-in service on this host with the Cloudflare Turnstile token in the x-captcha-response header. The service
// checks the email again and answers {status: "received"}, also for an email it already has.
;(() => {
  const messages = {
    sent: 'Thank you. We got your application, and we write to you at this address when your access is ready.',
    email: 'Enter a valid email address.',
    useCase: 'Tell us what you want to run.',
    check: 'Wait for the spam check to finish, then try again.',
    checkFailed: 'The spam check did not pass. Try again.',
    noMailServer: 'This email domain cannot receive email. Check the address.',
    invalid: 'Check the email address and the other fields, then try again.',
    tooMany: 'Too many tries. Wait a minute, then try again.',
    failed: 'We could not save your application. Try again in a minute.',
  }

  function failure(status, code) {
    if (status === 429) return messages.tooMany
    if (code === 'NO_MAIL_SERVER') return messages.noMailServer
    if (code === 'MISSING_RESPONSE' || code === 'VERIFICATION_FAILED') return messages.checkFailed
    if (status === 400) return messages.invalid
    return messages.failed
  }

  for (const form of document.querySelectorAll('form[data-waitlist]')) {
    const status = form.querySelector('[data-status]')
    const submit = form.querySelector('[data-submit]')
    const say = (text) => {
      status.textContent = text
    }

    form.addEventListener('submit', async (event) => {
      event.preventDefault()
      const data = new FormData(form)
      const email = form.elements.email
      if (!email.checkValidity() || !String(data.get('email')).includes('.')) {
        say(messages.email)
        email.focus()
        return
      }
      const useCase = form.elements.useCase
      if (useCase && !useCase.value.trim()) {
        say(messages.useCase)
        useCase.focus()
        return
      }
      const token = String(data.get('cf-turnstile-response') ?? '')
      if (!token) {
        say(messages.check)
        return
      }
      const body = { kind: form.dataset.waitlist }
      for (const name of ['email', 'name', 'company', 'useCase', 'volume']) {
        const value = String(data.get(name) ?? '').trim()
        if (value) body[name] = value
      }
      submit.disabled = true
      say('')
      try {
        const response = await fetch(form.dataset.endpoint, {
          method: 'POST',
          headers: { 'content-type': 'application/json', 'x-captcha-response': token },
          body: JSON.stringify(body),
        })
        if (response.ok) {
          form.reset()
          say(messages.sent)
          return
        }
        const answer = await response.json().catch(() => null)
        say(failure(response.status, answer && answer.code))
      } catch {
        say(messages.failed)
      } finally {
        submit.disabled = false
        // A Turnstile token works once, so each send needs a new check.
        window.turnstile?.reset(form.querySelector('.cf-turnstile'))
      }
    })
  }
})()
