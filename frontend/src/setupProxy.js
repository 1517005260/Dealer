const proxy = require('http-proxy-middleware')

module.exports = function (app) {
  app.use(
    proxy('/dealer', {
      target: 'http://localhost:8000',
      pathRewrite: {'^/dealer': '' }
    }),

  )
}