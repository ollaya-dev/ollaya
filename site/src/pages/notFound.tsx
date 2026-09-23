import { LogoMark } from '../components/Logo'

function ErrorPage({ code, message, detail }: { code: number; message: string; detail: string }) {
  return (
    <div class="mx-auto flex max-w-2xl flex-col items-start px-4 pt-16 md:px-6 md:pt-28">
      <LogoMark class="size-14 text-fg" />
      <h1 class="mt-8 text-[28px] font-medium tracking-tight text-fg">
        {code}. <span class="text-muted">{message}</span>
      </h1>
      <p class="mt-3 text-body">{detail}</p>
      <p class="mt-8 flex flex-wrap gap-x-5 gap-y-2 text-sm">
        <a href="/" class="font-medium text-fg underline-offset-4 hover:underline">
          Home
        </a>
        <a href="/search" class="font-medium text-fg underline-offset-4 hover:underline">
          Models
        </a>
        <a href="/docs" class="font-medium text-fg underline-offset-4 hover:underline">
          Docs
        </a>
      </p>
    </div>
  )
}

/** Rendered to dist/404.html (served with status 404 by Cloudflare and GitHub Pages). */
export function NotFoundPage() {
  return <ErrorPage code={404} message="That’s an error." detail="The page was not found." />
}
