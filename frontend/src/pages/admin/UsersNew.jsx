import { useState, useEffect } from 'react'
import AdminLayout from '../../components/layouts/AdminLayout'
import { adminAPI } from '../../services/api'
import toast from 'react-hot-toast'
import {
  Users, Loader, Search, Shield, UserX, UserCheck, Mail, Calendar, ChevronDown
} from 'lucide-react'

const AdminUsers = () => {
  const [users, setUsers] = useState([])
  const [loading, setLoading] = useState(true)
  const [searchQuery, setSearchQuery] = useState('')
  const [roleFilter, setRoleFilter] = useState('')

  useEffect(() => { fetchUsers() }, [roleFilter])

  const fetchUsers = async () => {
    setLoading(true)
    try {
      const res = await adminAPI.getUsers(roleFilter || null)
      setUsers(res.data)
    } catch { toast.error('Failed to fetch users') }
    finally { setLoading(false) }
  }

  const handleDeactivate = async (id) => {
    if (!confirm('Deactivate this user?')) return
    try {
      await adminAPI.deactivateUser(id)
      toast.success('User deactivated')
      fetchUsers()
    } catch { toast.error('Failed to deactivate user') }
  }

  const filtered = users.filter(u =>
    (u.email || '').toLowerCase().includes(searchQuery.toLowerCase()) ||
    (u.full_name || '').toLowerCase().includes(searchQuery.toLowerCase())
  )

  return (
    <AdminLayout title="User Management" subtitle="Manage platform users and roles">
      <div className="space-y-6">
        <div className="flex flex-col sm:flex-row gap-3">
          <div className="relative flex-1">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 h-4 w-4 text-surface-400" />
            <input type="text" value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)}
              className="input-field pl-11" placeholder="Search users..." />
          </div>
          <select value={roleFilter} onChange={(e) => setRoleFilter(e.target.value)} className="input-field w-auto">
            <option value="">All Roles</option>
            <option value="admin">Admin</option>
            <option value="landlord">Landlord</option>
            <option value="renter">Renter</option>
          </select>
        </div>

        {loading ? (
          <div className="flex justify-center py-16"><Loader className="h-8 w-8 animate-spin text-primary-600" /></div>
        ) : filtered.length > 0 ? (
          <div className="card p-0 overflow-hidden">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-surface-200 dark:border-surface-700 bg-surface-50 dark:bg-surface-800">
                    {['User', 'Role', 'Plan', 'Scans', 'Joined', 'Status', 'Actions'].map(h => (
                      <th key={h} className="text-left py-3 px-4 text-xs font-medium text-surface-500 dark:text-surface-400 uppercase tracking-wider">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {filtered.map((user) => (
                    <tr key={user.id} className="border-b border-surface-100 dark:border-surface-700/50 hover:bg-surface-50 dark:hover:bg-surface-700/30 transition-colors">
                      <td className="py-3 px-4">
                        <div className="flex items-center gap-3">
                          <div className="w-8 h-8 bg-primary-100 dark:bg-primary-900/30 rounded-lg flex items-center justify-center text-primary-600 font-bold text-xs">
                            {(user.full_name || user.email || '?')[0].toUpperCase()}
                          </div>
                          <div>
                            <p className="text-sm font-medium text-surface-900 dark:text-white">{user.full_name || 'N/A'}</p>
                            <p className="text-xs text-surface-500 dark:text-surface-400">{user.email}</p>
                          </div>
                        </div>
                      </td>
                      <td className="py-3 px-4">
                        <span className={`badge ${user.role === 'admin' ? 'badge-danger' : user.role === 'landlord' ? 'badge-info' : 'badge-success'}`}>
                          {user.role}
                        </span>
                      </td>
                      <td className="py-3 px-4 capitalize text-surface-700 dark:text-surface-300">{user.subscription_plan || '-'}</td>
                      <td className="py-3 px-4 text-surface-700 dark:text-surface-300">{user.scans_remaining ?? '-'}</td>
                      <td className="py-3 px-4 text-surface-500 dark:text-surface-400 text-xs">{user.created_at ? new Date(user.created_at).toLocaleDateString() : '-'}</td>
                      <td className="py-3 px-4">
                        <span className={`badge ${user.is_active !== false ? 'badge-success' : 'badge-danger'}`}>
                          {user.is_active !== false ? 'Active' : 'Inactive'}
                        </span>
                      </td>
                      <td className="py-3 px-4">
                        {user.role !== 'admin' && user.is_active !== false && (
                          <button onClick={() => handleDeactivate(user.id)} className="btn btn-sm btn-ghost text-red-500">
                            <UserX className="h-3.5 w-3.5" />
                          </button>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : (
          <div className="card text-center py-16">
            <Users className="h-16 w-16 text-surface-300 mx-auto mb-4" />
            <h3 className="text-lg font-semibold text-surface-600 mb-2">No users found</h3>
          </div>
        )}
      </div>
    </AdminLayout>
  )
}

export default AdminUsers
