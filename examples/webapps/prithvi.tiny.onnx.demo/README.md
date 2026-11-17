# Prithvi.tiny Web Demo

This is a PWA (Progressive Web App) demo to run a fine-tuned Prithvi.tiny model as ONNX withing a browser/mobile device. It uses the WASM (Web Assembly) ONNX execution engine.

The model was fine-tuned using the Chesapeake Bay Land Use/Land Cover and Change Dataset: https://www.chesapeakeconservancy.org/projects/cbp-land-use-land-cover-data-project

Copy the example images to this directory and to dist:

```
wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_3_rgb.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_3_x.bin

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_3_y.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_3_pred_mask.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_16_rgb.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_16_x.bin

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_16_y.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_16_pred_mask.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_18_rgb.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_18_x.bin

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_18_y.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_18_pred_mask.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_21_rgb.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_21_x.bin

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_21_y.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_21_pred_mask.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_23_rgb.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_23_x.bin

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_23_y.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_23_pred_mask.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_37_rgb.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_37_x.bin

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_37_y.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_37_pred_mask.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_39_rgb.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_39_x.bin

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_39_y.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_39_pred_mask.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_40_rgb.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_40_x.bin

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_40_y.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_40_pred_mask.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_42_rgb.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_42_x.bin

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_42_y.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_42_pred_mask.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_49_rgb.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_49_x.bin

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_49_y.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_49_pred_mask.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_51_rgb.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_51_x.bin

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_51_y.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_51_pred_mask.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_80_rgb.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_80_x.bin

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_80_y.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_80_pred_mask.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_108_rgb.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_108_x.bin

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_108_y.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_108_pred_mask.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_111_rgb.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_111_x.bin

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_111_y.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_111_pred_mask.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_112_rgb.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_112_x.bin

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_112_y.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_112_pred_mask.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_113_rgb.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_113_x.bin

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_113_y.png

wget https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/sample_113_pred_mask.png
```

Just copy the contents of the /dist folder to a web server/servable COS bucket of your choise. Or simply run 'python3 -m http.server' from the /dist folder to test locally.

Please also copy the ONNX model to the /dist folder before. You can get it from: https://prithvi-demo-onnx-mobile.s3-web.eu-de.cloud-object-storage.appdomain.cloud/model_chesapeake.onnx

## Developing

This app is built on [Carbon Web Components](https://carbondesignsystem.com/developing/frameworks/web-components/) using [Vite](https://vite.dev/).

1. Install all packages using `pnpm i`
2. Install PWA with vite using `pnpm add -D vite-plugin-pwa`
3. Run locally to test changes using `pnpm dev`
4. Build changes into the `dist` folder using `pnpm build`
5. From the `dist` folder, run `python3 -m http.server`

_Note: Always make sure to clear cookies and cache before running this app. It can help to open in a private browser._
