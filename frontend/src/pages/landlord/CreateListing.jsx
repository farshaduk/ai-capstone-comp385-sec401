import { useState } from 'react'
import LandlordLayout from '../../components/layouts/LandlordLayout'
import { landlordAPI } from '../../services/api'
import { Plus, Loader, MapPin, DollarSign, Home, Bed, Bath } from 'lucide-react'
import toast from 'react-hot-toast'
import { useNavigate } from 'react-router-dom'

const initialForm = {
  title: '', address: '', city: 'Toronto', province: 'ON', postal_code: '',
  price: '', beds: 1, baths: 1, sqft: '', property_type: 'apartment',
  description: '', amenities: [], laundry: 'in_unit', utilities: 'not_included',
  pet_friendly: false, parking_included: false, available_date: '',
}

const amenityOptions = ['Gym', 'Pool', 'Rooftop', 'Concierge', 'Balcony', 'Dishwasher', 'AC', 'Storage', 'Bike Room']

const CreateListing = () => {
  const [form, setForm] = useState(initialForm)
  const [submitting, setSubmitting] = useState(false)
  const navigate = useNavigate()

  const set = (field) => (e) => setForm(prev => ({ ...prev, [field]: e.target.value }))
  const toggle = (field) => () => setForm(prev => ({ ...prev, [field]: !prev[field] }))
  const toggleAmenity = (a) => {
    setForm(prev => ({
      ...prev,
      amenities: prev.amenities.includes(a) ? prev.amenities.filter(x => x !== a) : [...prev.amenities, a]
    }))
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!form.title.trim() || !form.address.trim() || !form.price) {
      return toast.error('Please fill in title, address, and price')
    }
    setSubmitting(true)
    try {
      await landlordAPI.createListing({
        ...form,
        price: parseFloat(form.price),
        beds: parseInt(form.beds),
        baths: parseFloat(form.baths),
        sqft: form.sqft ? parseInt(form.sqft) : null,
      })
      toast.success('Listing submitted for admin review! It will be visible to tenants once approved.')
      navigate('/landlord/my-listings')
    } catch (err) {
      console.error(err)
      toast.error('Failed to create listing')
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <LandlordLayout title="Create Listing" subtitle="Add a new rental property listing">
      <form onSubmit={handleSubmit} className="max-w-3xl mx-auto space-y-6">
        {/* Basic Info */}
        <div className="card p-6">
          <h3 className="font-semibold text-surface-900 dark:text-white mb-4 flex items-center gap-2">
            <Home className="h-5 w-5 text-landlord-500" /> Property Details
          </h3>
          <div className="space-y-4">
            <div>
              <label className="input-label">Listing Title *</label>
              <input type="text" value={form.title} onChange={set('title')} className="input-field" placeholder="e.g. Spacious 2BR in Liberty Village" required />
            </div>
            <div className="grid md:grid-cols-2 gap-4">
              <div>
                <label className="input-label">Street Address *</label>
                <input type="text" value={form.address} onChange={set('address')} className="input-field" placeholder="123 King St W" required />
              </div>
              <div>
                <label className="input-label">City</label>
                <input type="text" value={form.city} onChange={set('city')} className="input-field" />
              </div>
            </div>
            <div className="grid md:grid-cols-3 gap-4">
              <div>
                <label className="input-label">Province</label>
                <input type="text" value={form.province} onChange={set('province')} className="input-field" />
              </div>
              <div>
                <label className="input-label">Postal Code</label>
                <input type="text" value={form.postal_code} onChange={set('postal_code')} className="input-field" placeholder="M5V 1A1" maxLength={7} />
              </div>
              <div>
                <label className="input-label">Property Type</label>
                <select value={form.property_type} onChange={set('property_type')} className="input-field">
                  <option value="apartment">Apartment</option>
                  <option value="condo">Condo</option>
                  <option value="house">House</option>
                  <option value="townhouse">Townhouse</option>
                  <option value="basement">Basement</option>
                  <option value="room">Room</option>
                </select>
              </div>
            </div>
          </div>
        </div>

        {/* Pricing & Size */}
        <div className="card p-6">
          <h3 className="font-semibold text-surface-900 dark:text-white mb-4 flex items-center gap-2">
            <DollarSign className="h-5 w-5 text-landlord-500" /> Pricing & Size
          </h3>
          <div className="grid md:grid-cols-4 gap-4">
            <div>
              <label className="input-label">Monthly Rent *</label>
              <input type="number" value={form.price} onChange={set('price')} className="input-field" placeholder="2400" min="0" required />
            </div>
            <div>
              <label className="input-label">Bedrooms</label>
              <select value={form.beds} onChange={set('beds')} className="input-field">
                <option value="0">Studio</option><option value="1">1</option><option value="2">2</option><option value="3">3</option><option value="4">4+</option>
              </select>
            </div>
            <div>
              <label className="input-label">Bathrooms</label>
              <select value={form.baths} onChange={set('baths')} className="input-field">
                <option value="1">1</option><option value="1.5">1.5</option><option value="2">2</option><option value="2.5">2.5</option><option value="3">3+</option>
              </select>
            </div>
            <div>
              <label className="input-label">Sq Ft</label>
              <input type="number" value={form.sqft} onChange={set('sqft')} className="input-field" placeholder="850" min="0" />
            </div>
          </div>
        </div>

        {/* Features */}
        <div className="card p-6">
          <h3 className="font-semibold text-surface-900 dark:text-white mb-4">Features & Amenities</h3>
          <div className="grid md:grid-cols-3 gap-4 mb-4">
            <div>
              <label className="input-label">Laundry</label>
              <select value={form.laundry} onChange={set('laundry')} className="input-field">
                <option value="in_unit">In Unit</option><option value="shared">Shared</option><option value="none">None</option>
              </select>
            </div>
            <div>
              <label className="input-label">Utilities</label>
              <select value={form.utilities} onChange={set('utilities')} className="input-field">
                <option value="not_included">Not Included</option><option value="included">Included</option><option value="partial">Partial</option>
              </select>
            </div>
            <div>
              <label className="input-label">Available Date</label>
              <input type="date" value={form.available_date} onChange={set('available_date')} className="input-field" />
            </div>
          </div>
          <div className="flex flex-wrap gap-3 mb-4">
            <label className="flex items-center gap-2 cursor-pointer">
              <input type="checkbox" checked={form.pet_friendly} onChange={toggle('pet_friendly')} className="rounded" />
              <span className="text-sm">Pet Friendly</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input type="checkbox" checked={form.parking_included} onChange={toggle('parking_included')} className="rounded" />
              <span className="text-sm">Parking Included</span>
            </label>
          </div>
          <label className="input-label">Amenities</label>
          <div className="flex flex-wrap gap-2">
            {amenityOptions.map(a => (
              <button type="button" key={a} onClick={() => toggleAmenity(a)}
                className={`px-3 py-1.5 rounded-full text-xs font-medium border transition-colors ${
                  form.amenities.includes(a)
                    ? 'bg-landlord-100 text-landlord-700 border-landlord-300 dark:bg-landlord-900/30 dark:text-landlord-400 dark:border-landlord-700'
                    : 'bg-surface-50 text-surface-600 border-surface-200 dark:bg-surface-800 dark:text-surface-400 dark:border-surface-700'
                }`}>
                {a}
              </button>
            ))}
          </div>
        </div>

        {/* Description */}
        <div className="card p-6">
          <label className="input-label">Description</label>
          <textarea value={form.description} onChange={set('description')} rows={4} className="input-field"
            placeholder="Describe the property, neighborhood, and any unique features..." />
        </div>

        <button type="submit" disabled={submitting} className="btn btn-primary btn-lg w-full">
          {submitting ? <><Loader className="h-5 w-5 animate-spin" /> Creating...</> : <><Plus className="h-5 w-5" /> Create Listing</>}
        </button>
      </form>
    </LandlordLayout>
  )
}

export default CreateListing
