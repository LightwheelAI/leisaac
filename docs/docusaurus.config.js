// @ts-check
// `@type` JSDoc annotations allow editor autocompletion and type checking
// (when paired with `@ts-check`).
// There are various equivalent ways to declare your Docusaurus config.
// See: https://docusaurus.io/docs/api/docusaurus-config

import {themes as prismThemes} from 'prism-react-renderer';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'LeIsaac Document',
  tagline: 'LeIsaac Docs: Quickstart, Guides, and Resources',
  favicon: 'img/favicon.ico',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: 'https://lightwheelai.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/leisaac/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'LightwheelAI', // Usually your GitHub org/user name.
  projectName: 'leisaac', // Usually your repo name.

  onBrokenLinks: 'throw',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: './sidebars.js',
          routeBasePath: '/',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/lightwheelai/leisaac/tree/main/docs/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      // Replace with your project's social card
      image: 'img/docusaurus-social-card.jpg',
      colorMode: {
        respectPrefersColorScheme: true,
      },
      navbar: {
        title: 'LeIsaac',
        logo: {
          alt: 'leisaac Logo',
          src: 'img/logo.png',
        },
        items: [
          { type: 'docSidebar', sidebarId: 'docs', position: 'left', label: 'Docs' },
          { type: 'docSidebar', sidebarId: 'resources', position: 'left', label: 'Resources' },
          {
            href: 'https://github.com/LightwheelAI/leisaac',
            position: 'right',
            className: 'header-github-link',
            'aria-label': 'GitHub repository',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Outline',
            items: [
              {
                label: 'Docs',
                to: '/',
              },
              {
                label: 'Resources',
                to: '/resources/available_robots',
              },
            ],
          },
          {
            title: 'Community',
            items: [
              {
                label: 'Lightwheel AI',
                href: 'https://lightwheel.ai/',
              },
              {
                label: 'LinkedIn',
                href: 'https://www.linkedin.com/company/lightwheel-ai/',
              },
            ],
          },
          {
            title: 'More',
            items: [
              {
                label: 'GitHub',
                href: 'https://github.com/LightwheelAI/leisaac',
              },
            ],
          },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} Lightwheel AI, Inc. Built with Docusaurus.`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
        additionalLanguages: ['bash']
      },
    }),
};

export default config;
