// @vitest-environment jsdom

import { cleanup, render, screen } from '@testing-library/react'
import { createMemoryRouter, RouterProvider } from 'react-router-dom'
import { afterEach, describe, expect, it, vi } from 'vitest'
import App from '../src/App'

vi.mock('../src/pages/Dashboard', () => ({ default: () => <div>Dashboard page</div> }))
vi.mock('../src/pages/Sessions', () => ({ default: () => <div>Sessions page</div> }))
vi.mock('../src/pages/SessionDetail', () => ({ default: () => <div>Session detail page</div> }))
vi.mock('../src/pages/Runs', () => ({ default: () => <div>Runs page</div> }))
vi.mock('../src/pages/RunDetail', () => ({ default: () => <div>Run detail page</div> }))
vi.mock('../src/pages/Analyze', () => ({ default: () => <div>Analysis page</div> }))

afterEach(cleanup)

describe('application shell', () => {
  it('provides primary navigation and renders the requested route', () => {
    const router = createMemoryRouter(
      [{ path: '*', element: <App /> }],
      { initialEntries: ['/runs'] },
    )
    render(<RouterProvider router={router} />)

    expect(screen.getByRole('link', { name: 'Dashboard' }).getAttribute('href')).toBe('/')
    expect(screen.getByRole('link', { name: 'Sessions' }).getAttribute('href')).toBe('/sessions')
    expect(screen.getByRole('link', { name: 'Runs' }).getAttribute('href')).toBe('/runs')
    expect(screen.getByRole('link', { name: 'Analysis' }).getAttribute('href')).toBe('/analyze')
    expect(screen.getByRole('main').textContent).toContain('Runs page')
  })
})
