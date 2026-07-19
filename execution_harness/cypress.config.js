const { defineConfig } = require('cypress');

module.exports = defineConfig({
  e2e: {
    specPattern: 'cypress/e2e/**/*.cy.js',
    // Set per-run by the orchestrator so relative cy.visit('/...') calls
    // resolve against the story's source app.
    baseUrl: process.env.CYPRESS_BASE_URL || 'https://www.saucedemo.com',
    supportFile: false,
    video: false,
    screenshotOnRunFailure: false,
    defaultCommandTimeout: 10000,
    pageLoadTimeout: 60000,
  },
});
