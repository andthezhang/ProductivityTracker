## Host frontend application.
`cd frontend`
`yarn`
`yarn build` for production, `yarn watch` for  development

## Serve trained model locally (allow cors).
`npm install --global http-server`
`http-server backend/saved_models --cors=Authorization`