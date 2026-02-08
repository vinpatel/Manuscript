// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// https://astro.build/config
export default defineConfig({
	site: 'https://vinpatel.github.io',
	base: '/manuscript',
	integrations: [
		starlight({
			title: 'Manuscript',
			description: 'Open Source AI Content Detector - Privacy-First, Multi-Modal Detection',
			logo: {
				light: './src/assets/logo-light.svg',
				dark: './src/assets/logo-dark.svg',
				replacesTitle: false,
			},
			customCss: [
				'./src/styles/custom.css',
			],
			social: [
				{ icon: 'github', label: 'GitHub', href: 'https://github.com/vinpatel/manuscript' },
				{ icon: 'x.com', label: 'Twitter', href: 'https://twitter.com/manuscript' },
			],
			head: [
				{
					tag: 'meta',
					attrs: {
						property: 'og:image',
						content: 'https://vinpatel.github.io/manuscript/og-image.png',
					},
				},
				{
					tag: 'meta',
					attrs: {
						property: 'og:title',
						content: 'Manuscript - Open Source AI Content Detector',
					},
				},
				{
					tag: 'meta',
					attrs: {
						property: 'og:description',
						content: 'Detect AI-generated text, images, audio & video—100% offline, self-hosted, zero external calls.',
					},
				},
			],
			sidebar: [
				{
					label: 'Getting Started',
					items: [
						{ label: 'Introduction', slug: 'introduction' },
						{ label: 'Quick Start', slug: 'quickstart' },
						{ label: 'Installation', slug: 'installation' },
					],
				},
				{
					label: 'Detection',
					items: [
						{ label: 'Text Detection', slug: 'text-detection' },
						{ label: 'Image Detection', slug: 'image-detection' },
						{ label: 'Audio Detection', slug: 'audio-detection' },
						{ label: 'Video Detection', slug: 'video-detection' },
					],
				},
				{
					label: 'Benchmarks',
					items: [
						{ label: 'Overview', slug: 'benchmarks' },
						{ label: 'Methodology', slug: 'methodology' },
						{ label: 'Datasets', slug: 'datasets' },
					],
				},
				{
					label: 'API Reference',
					autogenerate: { directory: 'api' },
				},
			],
		}),
	],
});
