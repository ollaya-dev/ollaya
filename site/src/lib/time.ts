const long = new Intl.DateTimeFormat('en-US', { dateStyle: 'long', timeZone: 'UTC' })
const medium = new Intl.DateTimeFormat('en-US', { dateStyle: 'medium', timeZone: 'UTC' })

/** "September 23, 2026" */
export function formatDate(iso: string): string {
  return long.format(new Date(iso))
}

/** "Sep 23, 2026" */
export function formatShortDate(iso: string): string {
  return medium.format(new Date(iso))
}
