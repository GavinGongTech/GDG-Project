import { resolve } from 'path'
import { defineConfig } from 'vite'

export default defineConfig({
  root: 'Frontend',
  build: {
    outDir: '../dist',
    emptyOutDir: true,
    rollupOptions: {
      input: {
        main: resolve(__dirname, 'Frontend/index.html'),
        landing: resolve(__dirname, 'Frontend/landing_page.html'),
        predict: resolve(__dirname, 'Frontend/predict.html'),
        login: resolve(__dirname, 'Frontend/login_page.html'),
        team: resolve(__dirname, 'Frontend/team_page.html'),
        result: resolve(__dirname, 'Frontend/result_page.html')
      }
    }
  },
  server: {
    port: 3000,
    open: true
  }
})
