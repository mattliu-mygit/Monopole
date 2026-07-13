import { NavLink, Outlet, Route, Routes } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import Sessions from './pages/Sessions'
import SessionDetailPage from './pages/SessionDetail'
import Runs from './pages/Runs'
import RunDetail from './pages/RunDetail'
import Analyze from './pages/Analyze'
import Monitor from './pages/Monitor'
import Jobs from './pages/Jobs'

const navItems = [
  { to: '/', label: 'Dashboard' },
  { to: '/sessions', label: 'Sessions' },
  { to: '/runs', label: 'Runs' },
  { to: '/analyze', label: 'Analysis' },
  { to: '/monitor', label: 'Monitor' },
  { to: '/jobs', label: 'Jobs' },
]

function Layout() {
  return (
    <div className="flex h-screen">
      <nav className="w-56 bg-gray-900 text-white flex flex-col p-4 gap-1">
        <div className="text-sm font-semibold text-gray-400 uppercase tracking-wider mb-4">
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
      <main className="flex-1 overflow-auto bg-gray-50 p-6">
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
        <Route path="monitor" element={<Monitor />} />
        <Route path="jobs" element={<Jobs />} />
      </Route>
    </Routes>
  )
}
