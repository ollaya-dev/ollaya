**In a web page** (`npm install @ollaya/web`): the model downloads once ({{web_size}}) and runs on the device.

```js
import { fromHub, load } from '@ollaya/web';

const model = await load(fromHub('{{repo}}'));
const { answers } = await model.decide({
  state: "I was charged twice for order A-104. Refund the duplicate today or I'm cancelling.",
  questions: {
    refund: { type: 'noul', instructions: 'Is the customer asking for a refund?' },
    team: { type: 'choice', instructions: 'Which team handles this?',
            criteria: { billing: 'Payments, charges, refunds', technical: 'Bugs, outages, errors' } },
    urgency: { type: 'score', instructions: 'How urgent is this?', criteria: ['Can wait', 'This week', 'Today'] },
  },
});
```
