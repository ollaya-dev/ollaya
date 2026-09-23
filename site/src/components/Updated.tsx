import { formatDate, formatShortDate } from '../lib/time'
import { Icon } from './Icon'

/**
 * "Updated <date>". The page is static, so the date is rendered absolute and
 * public/static/app.js turns it into relative time ("today", "3 days ago") in the browser.
 */
export function Updated({ iso }: { iso: string }) {
  return (
    <span class="inline-flex items-center gap-1.5" title={formatDate(iso)}>
      <Icon name="clock" class="size-4" />
      <span>
        Updated{' '}
        <time datetime={iso} data-relative>
          {formatShortDate(iso)}
        </time>
      </span>
    </span>
  )
}
