import { NavLink } from "react-router-dom";

const navItems = [
  { to: "/", label: "Dashboard" },
  { to: "/sessions", label: "Sessions" },
  { to: "/runs", label: "Runs" },
  { to: "/analyze", label: "Analysis" },
  { to: "/monitor", label: "Monitor" },
  { to: "/jobs", label: "Jobs" },
];

export default function Sidebar() {
  return (
    <nav className="w-56 bg-gray-900 text-gray-300 min-h-screen p-4">
      <div className="text-lg font-bold text-white mb-6">Monopole</div>
      <ul className="space-y-1">
        {navItems.map(({ to, label }) => (
          <li key={to}>
            <NavLink
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                `block py-2 px-3 rounded text-sm ${
                  isActive
                    ? "bg-gray-700 text-white"
                    : "hover:text-white"
                }`
              }
            >
              {label}
            </NavLink>
          </li>
        ))}
      </ul>
    </nav>
  );
}
