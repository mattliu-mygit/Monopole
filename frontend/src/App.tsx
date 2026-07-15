import { NavLink, Outlet, Route, Routes } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import Sessions from './pages/Sessions'
import SessionDetailPage from './pages/SessionDetail'
import Runs from './pages/Runs'
import RunDetail from './pages/RunDetail'
import Analyze from './pages/Analyze'

const navItems = [
  { to: '/', label: 'Dashboard' },
  { to: '/sessions', label: 'Sessions' },
  { to: '/runs', label: 'Runs' },
  { to: '/analyze', label: 'Analysis' },
]

function Layout() {
  return (
    <div className="flex min-h-screen flex-col md:h-screen md:flex-row">
      <nav className="flex w-full shrink-0 flex-row flex-wrap gap-1 bg-gray-900 p-4 text-white md:w-56 md:flex-col md:flex-nowrap">
        <div className="mb-2 w-full text-sm font-semibold uppercase tracking-wider text-gray-400 md:mb-4">
          Weave Agent Signals
        </div>
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) =>
              `px-3 py-2 rounded text-sm ${
                isActive ? 'bg-gray-700 text-white' : 'text-gray-300 hover:bg-gray-800'
              }`
            }
          >
            {item.label}
          </NavLink>
        ))}
      </nav>
      <main className="min-w-0 flex-1 overflow-auto bg-gray-50 p-4 sm:p-6">
        <Outlet />
      </main>
    </div>
  )
}

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route index element={<Dashboard />} />
        <Route path="sessions" element={<Sessions />} />
        <Route path="sessions/:id" element={<SessionDetailPage />} />
        <Route path="runs" element={<Runs />} />
        <Route path="runs/:runId" element={<RunDetail />} />
        <Route path="analyze" element={<Analyze />} />
      </Route>
    </Routes>
  )
}
