// Its own module, so the mock backend can throw it without importing backend.ts (which imports the mock).

/** An Ollaya Cloud failure: `code` is INSUFFICIENT_CREDITS, SIGNED_OUT, NETWORK, a gateway code… */
export class CloudError extends Error {
  constructor(
    readonly code: string,
    message: string,
  ) {
    super(message)
  }
  override toString() {
    return this.message
  }
}
