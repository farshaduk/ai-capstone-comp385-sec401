import { Navigate, useLocation } from 'react-router-dom'
import { useAuthStore } from '../../store/authStore'

/**
 * RoleRoute — Strict role-based route guard
 * Prevents dashboard leakage between roles.
 * If user role doesn't match allowed roles, redirect to their correct dashboard.
 */
const RoleRoute = ({ children, allowedRoles }) => {
  const { isAuthenticated, user } = useAuthStore()
  const location = useLocation()

  if (!isAuthenticated || !user) {
    return <Navigate to="/login" state={{ from: location }} replace />
  }

  const userRole = user.role

  // Normalize role names
  const normalizedAllowed = allowedRoles.map(r => r === 'tenant' ? 'renter' : r)
  const isAllowed = normalizedAllowed.includes(userRole)

  if (!isAllowed) {
    // Redirect to their OWN dashboard — no leakage
    const redirectMap = {
      admin: '/admin',
      landlord: '/landlord',
      renter: '/tenant',
      tenant: '/tenant',
    }
    return <Navigate to={redirectMap[userRole] || '/'} replace />
  }

  return children
}

export default RoleRoute
