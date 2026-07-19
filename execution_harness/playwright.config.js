const { defineConfig } = require('@playwright/test');

module.exports = defineConfig({
  testDir: './pw_specs',
  timeout: 90000,
  retries: 0,
  workers: 1,
  use: {
    // Set per-run by the orchestrator so relative page.goto('/...') calls
    // resolve against the story's source app.
    baseURL: process.env.PW_BASE_URL || 'https://www.saucedemo.com',
    headless: true,
    actionTimeout: 15000,
    navigationTimeout: 60000,
  },
});
