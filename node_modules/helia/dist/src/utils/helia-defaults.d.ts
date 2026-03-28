/**
 * @packageDocumentation
 *
 * Exports a `createHelia` function that returns an object that implements the {@link Helia} API.
 *
 * Pass it to other modules like {@link https://www.npmjs.com/package/@helia/unixfs | @helia/unixfs} to make files available on the distributed web.
 *
 * @example
 *
 * ```typescript
 * import { createHelia } from 'helia'
 * import { unixfs } from '@helia/unixfs'
 * import { CID } from 'multiformats/cid'
 *
 * const helia = await createHelia()
 *
 * const fs = unixfs(helia)
 * fs.cat(CID.parse('bafyFoo'))
 * ```
 */
import type { HeliaInit } from '@helia/utils';
import type { Libp2p } from '@libp2p/interface';
/**
 * Create and return the default options used to create a Helia node
 *
 * @example Adding an additional libp2p service
 *
 * ```ts
 * import { myService } from '@example/my-service'
 * import { createHelia, heliaDefaults } from 'helia'
 *
 * // get a copy of the default libp2p config
 * const init = heliaDefaults()
 *
 * // add the custom service to the service map
 * init.libp2p.services.myService = myService()
 *
 * // create a Helia node with the custom config
 * const helia = await createHelia(init)
 *
 * //... use service
 * helia.libp2p.services.myService.serviceMethod()
 * ```
 */
export declare function heliaDefaults<T extends Libp2p>(init?: Partial<HeliaInit<T>>): Promise<Omit<HeliaInit<T>, 'libp2p'> & {
    libp2p: T;
}>;
//# sourceMappingURL=helia-defaults.d.ts.map