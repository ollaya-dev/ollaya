/**
 * The /try demo: the model it loads in the browser and the ready examples. The page renders both;
 * src/client/try.js reads them back from the page (the model from #try-app's data attributes, the
 * examples from #try-examples). Query parameters override the model for testing:
 * ?repo=<org/name>&revision=<commit>&endpoint=<url>.
 */

// `revision` is the repository's commit: the browser caches files at a commit, so a returning visitor loads
// without downloading.
export const MODEL = {
  repo: 'ollaya-dev/coba-150m-ternary',
  revision: '3609cf41de228721ac41b0bae70a6c26e7c19f17',
  name: 'coba',
  tag: '150m-ternary',
  size: '66 MB',
}

export type TryQuestion =
  | { type: 'noul'; instructions: string }
  | { type: 'choice'; instructions: string; criteria: Record<string, string> }
  | { type: 'score'; instructions: string; criteria: string[] }

export interface TryExample {
  key: string
  label: string
  state: string
  questions: Record<string, TryQuestion>
}

export const EXAMPLES: TryExample[] = [
  {
    key: 'support',
    label: 'Support triage',
    state:
      "Hi, I was charged twice for my Pro plan this month (invoice #4471). Please refund the duplicate before Friday, or I'll have to cancel. Dana",
    questions: {
      refund: { type: 'noul', instructions: 'The customer asks for money back.' },
      team: {
        type: 'choice',
        instructions: 'Which team should handle this ticket?',
        criteria: {
          billing: 'Payments, invoices and refunds',
          technical: 'Bugs, errors and outages',
          account: 'Login, profile and settings',
        },
      },
      urgency: {
        type: 'score',
        instructions: 'How urgent is this ticket?',
        criteria: ['Can wait', 'Needs attention this week', 'Needs attention today'],
      },
    },
  },
  {
    key: 'moderation',
    label: 'Moderation',
    state: "Honestly the dumbest take I've read all week. People like you shouldn't be allowed anywhere near a keyboard.",
    questions: {
      toxic: { type: 'noul', instructions: 'The comment insults or attacks someone.' },
      category: {
        type: 'choice',
        instructions: 'What kind of comment is this?',
        criteria: {
          harassment: 'Insults or attacks a person',
          spam: 'Ads, links or repeated promotion',
          criticism: 'Disagrees with the content, without attacking anyone',
          friendly: 'Supportive or neutral',
        },
      },
    },
  },
  {
    key: 'sales',
    label: 'Lead qualification',
    state:
      "We're a 120-person logistics company looking for a way to route inbound support email. Budget is approved for Q1, and I'd like a demo next week with our CTO.",
    questions: {
      qualified: { type: 'noul', instructions: 'This lead is ready for a sales call.' },
      segment: {
        type: 'choice',
        instructions: 'How large is the company?',
        criteria: { small: 'Fewer than 50 people', mid_market: '50 to 1,000 people', enterprise: 'More than 1,000 people' },
      },
      intent: {
        type: 'score',
        instructions: 'How close is the lead to buying?',
        criteria: ['Just browsing', 'Researching options', 'Comparing vendors', 'Ready to buy'],
      },
    },
  },
  {
    key: 'review',
    label: 'Product review',
    state:
      'Battery lasts two full days and the screen is gorgeous, but the camera struggles in low light. For the price I would still buy it again.',
    questions: {
      recommends: { type: 'noul', instructions: 'The reviewer would recommend the product.' },
      sentiment: {
        type: 'choice',
        instructions: 'What is the overall sentiment?',
        criteria: { positive: 'Mostly happy', mixed: 'Both praise and complaints', negative: 'Mostly unhappy' },
      },
      rating: {
        type: 'score',
        instructions: 'How many stars would the reviewer give?',
        criteria: ['1 star', '2 stars', '3 stars', '4 stars', '5 stars'],
      },
    },
  },
  {
    key: 'guard',
    label: 'Agent guard',
    state:
      'The user asked: "Clean up the build artifacts in my project." The agent wants to run the shell command: rm -rf ~/*',
    questions: {
      safe: { type: 'noul', instructions: 'This tool call is safe to run without asking the user.' },
      risk: {
        type: 'choice',
        instructions: 'What is the main risk of running it?',
        criteria: {
          data_loss: 'Deletes or overwrites files the user needs',
          data_leak: 'Sends private data somewhere',
          cost: 'Spends money or paid resources',
          none: 'No real risk',
        },
      },
    },
  },
]
