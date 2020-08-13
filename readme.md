## Generate Data



## Train Model (Offline)

Convert to tfjs model: 

`cd backend/saved_model/`

`tensorflowjs_converter --input_format=tf_saved_model --output_format tfjs_graph_model no_pose_model no_pose_model_tfjs` \

## Serve trained model locally (allow cors).

`npm install --global http-server`\
`http-server backend/saved_models --cors=Authorization`

## Host frontend application.

`cd frontend`\
`yarn`\
`yarn build` for production, `yarn watch` for  development

