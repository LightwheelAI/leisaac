// @ts-check

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.

 @type {import('@docusaurus/plugin-content-docs').SidebarsConfig}
 */
const sidebars = {
  docs: [
    'docs/introduction',
    {
      type: 'category',
      label: 'Getting Started',
      link: {
        type: 'generated-index',
        slug: '/docs/getting-started',
        title: 'Getting Started',
        description: 'LeIsaac getting started overview and quick guide.',
      },
      items: [
        {
          type: 'category',
          label: 'Installation',
          link: { type: 'doc', id: 'docs/getting_started/installation' },
          items: [],
        },
        {
          type: 'category',
          label: 'Teleoperation',
          link: { type: 'doc', id: 'docs/getting_started/teleoperation' },
          items: [],
        },
        {
          type: 'category',
          label: 'Dataset Replay',
          link: { type: 'doc', id: 'docs/getting_started/dataset_replay' },
          items: [],
        },
        {
          type: 'category',
          label: 'Policy Training & Inference',
          link: { type: 'doc', id: 'docs/getting_started/policy_support' },
          items: [],
        },
      ],
    },
    {
      type: 'category',
      label: 'Extra Features',
      link: {
        type: 'generated-index',
        slug: '/docs/features',
        title: 'Extra Features',
        description: 'We also provide some additional features. You can refer to the following instructions to try them out.'
      },
      items: [
        {
          type: 'category',
          label: 'DigitalTwin Env',
          link: { type: 'doc', id: 'docs/features/digitaltwin_env' },
          items: [],
        },
        {
          type: 'category',
          label: 'MimicGen Env',
          link: { type: 'doc', id: 'docs/features/mimicgen_env' },
          items: [],
        },
        {
          type: 'category',
          label: 'EnvHub Support',
          link: { type: 'doc', id: 'docs/features/envhub_support' },
          items: [],
        },
      ],
    },
    'docs/trouble_shooting',
    {
      type: 'category',
      label: 'Cloud Simulation',
      link: {
        type: 'generated-index',
        slug: '/docs/cloud_simulation',
        title: 'Cloud Simulation',
        description: 'Using LeIsaac on cloud platforms.'
      },
      items: [
        {
          type: 'category',
          label: 'NVIDIA Brev',
          link: { type: 'doc', id: 'docs/cloud_simulation/nvidia_brev' },
          items: [],
        },
      ],
    },
  ],

  resources: [
    'resources/available_robots',
    'resources/available_env',
    'resources/available_devices',
    'resources/available_policy',
  ],
};

export default sidebars;
