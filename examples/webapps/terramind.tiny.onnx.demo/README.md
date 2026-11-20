# TerraMind.tiny Web Demo

This is a PWA (Progressive Web App) demo to run a fine-tuned TerraMind.tiny model as ONNX withing a browser/mobile device. It uses the WASM (Web Assembly) ONNX execution engine.

The model was fine-tuned using the The Aerial Elephant Dataset: https://zenodo.org/records/3234780

Copy the example images to this directory and to dist:

```
wget https://terramind-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/tile_0_0_0.png

wget https://terramind-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/tile_0_1152_3072.png

wget https://terramind-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/tile_0_768_3072.png

wget https://terramind-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/tile_1260_0_2688.png

wget https://terramind-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/tile_100_3840_1536.png

wget https://terramind-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/tile_100_4224_1152.png

wget https://terramind-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/tile_100_4224_1536.png

wget https://terramind-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/tile_100_4224_768.png

wget https://terramind-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/tile_100_4608_0.png

wget https://terramind-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/tile_1261_4224_768.png

wget https://terramind-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/tile_1013_3072_3072.png

wget https://terramind-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/tile_101_3072_384.png
```

Just copy the contents of the /dist folder to a web server/servable COS bucket of your choise. Or simply run 'python3 -m http.server' from the /dist folder to test locally.

Please also copy the ONNX model to the /dist folder before. You can get it from: https://terramind-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/model_tm_tiny_aed_elephants.onnx

## Developing

This app is built on [Carbon Web Components](https://carbondesignsystem.com/developing/frameworks/web-components/) using [Vite](https://vite.dev/).

1. Install all packages using `pnpm i`
2. Install PWA with vite using `pnpm add -D vite-plugin-pwa`
3. Run locally to test changes using `pnpm dev`
4. Build changes into the `dist` folder using `pnpm build`
5. From the `dist` folder, run `python3 -m http.server`

_Note: Always make sure to clear cookies and cache before running this app. It can help to open in a private browser._
