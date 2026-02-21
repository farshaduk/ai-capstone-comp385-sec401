import { Navigate } from 'react-router-dom'
import { useAuthStore } from '../../store/authStore'

/**
 * PublicRoute — Redirect authenticated users away from public pages
 * Sends them to their role-specific dashboard
 */
const PublicRoute = ({ children }) => {
  const { isAuthenticated, user } = useAuthStore()

  if (isAuthenticated && user) {
    const redirectMap = {
      admin: '/admin',
      landlord: '/landlord',
      renter: '/tenant',
      tenant: '/tenant',
    }
    return <Navigate to={redirectMap[user.role] || '/tenant'} replace />
  }

  return children
}

export default PublicRoute
