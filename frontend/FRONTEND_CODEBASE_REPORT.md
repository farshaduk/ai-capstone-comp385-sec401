# RentalGuard Frontend — Comprehensive Codebase Report

> **Generated**: Deep-dive analysis of `frontend/src/`  
> **Stack**: React 18 · Zustand · Axios · Tailwind CSS · Vite · Vitest  
> **Total Files Analyzed**: 65+ source files  
> **Lines of Code (approx)**: ~12,500 LOC  

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Technology Stack](#2-technology-stack)
3. [File-by-File Analysis](#3-file-by-file-analysis)
   - [Services](#31-services)
   - [Stores](#32-stores)
   - [Guards](#33-guards)
   - [Layouts](#34-layouts)
   - [UI Components](#35-ui-components)
   - [Public Components](#36-public-components)
   - [Auth Pages](#37-auth-pages)
   - [Public Pages](#38-public-pages)
   - [Tenant Pages](#39-tenant-pages)
   - [Admin Pages (New — Active)](#310-admin-pages-new--active)
   - [Admin Pages (Old — Dead Code)](#311-admin-pages-old--dead-code)
   - [Landlord Pages (New — Active)](#312-landlord-pages-new--active)
   - [Landlord Pages (Old — Dead Code)](#313-landlord-pages-old--dead-code)
   - [Renter Pages (Legacy Duplicates)](#314-renter-pages-legacy-duplicates)
   - [CSS](#315-css)
   - [Test Files](#316-test-files)
   - [Routing (App.jsx)](#317-routing-appjsx)
4. [Design System](#4-design-system)
5. [Issues & Recommendations](#5-issues--recommendations)
6. [Dead Code Inventory](#6-dead-code-inventory)
7. [Summary Statistics](#7-summary-statistics)

---

## 1. Architecture Overview

```
src/
├── services/           # API client (Axios)
│   └── api.js
├── store/              # Zustand state management
│   ├── authStore.js
│   └── themeStore.js
├── guards/             # Route protection (auth + role)
│   ├── PublicRoute.jsx
│   └── RoleRoute.jsx
├── components/
│   ├── layouts/        # Role-based shell layouts (sidebar + topbar)
│   │   ├── AdminLayout.jsx
│   │   ├── LandlordLayout.jsx
│   │   └── TenantLayout.jsx
│   ├── ui/             # Shared micro-components
│   │   ├── LoadingSpinner.jsx
│   │   ├── Logo.jsx
│   │   └── ThemeToggle.jsx
│   ├── public/         # Landing page components
│   │   ├── PublicNavbar.jsx
│   │   └── Footer.jsx
│   └── Layout.jsx      # Legacy layout (unused in routing)
├── pages/
│   ├── Login.jsx
│   ├── Register.jsx
│   ├── public/         # Landing, GetStarted
│   ├── tenant/         # 11 tenant pages (active)
│   ├── admin/          # 7 new + 7 old pages
│   ├── landlord/       # 14 new + 5 old pages
│   └── renter/         # 4 legacy pages (dead code)
├── styles/
│   └── globals.css     # Tailwind config + design tokens
├── App.jsx             # Router (36 routes + 4 redirects)
└── __tests__/          # Vitest test suite (4 files)
```

**Key architectural patterns:**
- SPA with `BrowserRouter` (react-router-dom v6)
- Three role-based portals: Admin, Landlord, Tenant
- Each portal has its own `*Layout.jsx` (sidebar + topbar shell)
- Centralized API client with Bearer token injection
- Dual-version pattern: old pages (using `Layout`) and new pages (using role-specific layouts). **Only "New" versions are actively routed.**

---

## 2. Technology Stack

| Category | Technology | Details |
|----------|-----------|---------|
| Framework | React 18+ | Functional components, hooks only |
| Routing | react-router-dom v6 | BrowserRouter, Navigate, useSearchParams |
| State | Zustand | `persist` middleware; 2 stores (auth, theme) |
| HTTP | Axios | Interceptors for auth; centralized instance |
| Styling | Tailwind CSS | Custom design system with 280-line globals.css |
| Icons | Lucide React | Used in every single component |
| Charts | Recharts | BarChart, PieChart (admin/renter dashboards) |
| Notifications | React Hot Toast | `toast.success()` / `toast.error()` everywhere |
| Build | Vite | vite.config.js at project root |
| Testing | Vitest | @testing-library/react, jsdom environment |
| Fonts | Google Fonts | Inter (body), Plus Jakarta Sans (display), JetBrains Mono (code) |
| Dark Mode | Class strategy | Toggles `dark` class on `<html>`, system preference detection |

---

## 3. File-by-File Analysis

### 3.1 Services

#### `services/api.js` — ~210 lines
- **Purpose**: Centralized Axios HTTP client for all API communication
- **Key Features**:
  - Base URL: `http://localhost:8000/api` (hardcoded)
  - Request interceptor: injects `Authorization: Bearer <token>` from authStore
  - Response interceptor: auto-logout on 401 (clears authStore, redirects to `/login`)
  - Exports 5 API namespaces:
    - `authAPI` — login, register, getMe
    - `adminAPI` — 30+ methods (dashboard, datasets CRUD, model training, users, audit logs, feedback, auto-learning, subscription plans, BERT training/status/test, AI engines, preprocessing, message analysis, cross-document test, training data stats)
    - `renterAPI` — 20+ methods (analyze listing/URL/images/messages/conversations, history, subscription, feedback, export, listings, saved listings, applications, address verification, explainText)
    - `landlordAPI` — 20+ methods (document verification upload/base64, tenant verification, property image verification, cross-document, full application, listings CRUD, applicants, leases, analytics, messages/contacts/conversations, dashboard stats, verification history)
    - `profileAPI` — getProfile, updateProfile, changePassword
- **UI Patterns**: N/A (service layer)
- **Issues**:
  - ⚠️ Hardcoded `API_URL` — should use `import.meta.env.VITE_API_URL`
  - ⚠️ No retry logic or request queuing
  - ⚠️ Duplicate aliases: `runAutoLearning` vs `triggerAutoLearn` point to the same endpoint

---

### 3.2 Stores

#### `store/authStore.js` — ~45 lines
- **Purpose**: Authentication state management
- **Key Features**:
  - Persisted to `localStorage` via Zustand `persist`
  - State: `token`, `user`, `isAuthenticated`, `selectedRole`
  - Actions: `login(token, user)`, `logout()`, `setSelectedRole(role)`
  - Role helpers: `isAdmin()`, `isLandlord()`, `isTenant()` (checks both 'renter' and 'tenant')
  - `getDashboardPath()` — returns `/admin`, `/landlord`, or `/tenant`
- **Issues**:
  - ⚠️ `selectedRole` can be overwritten on login even if user has a different actual role
  - ⚠️ No token expiry / refresh token handling

#### `store/themeStore.js` — ~55 lines
- **Purpose**: Theme (light/dark/system) management
- **Key Features**:
  - Persisted via Zustand `persist` (only `theme` field, not `resolvedTheme`)
  - Three modes: `light`, `dark`, `system`
  - `applyTheme()` — toggles `dark` class on `document.documentElement`
  - `initTheme()` — sets up `prefers-color-scheme: dark` media query listener for system mode
- **Issues**: None significant

---

### 3.3 Guards

#### `guards/PublicRoute.jsx` — 26 lines
- **Purpose**: Redirect authenticated users away from public pages
- **Key Features**: Checks `isAuthenticated` from authStore; redirects to role-appropriate dashboard
- **UI Patterns**: Wraps children; transparent guard component
- **Issues**: None

#### `guards/RoleRoute.jsx` — 37 lines
- **Purpose**: Strict role-based access control
- **Key Features**:
  - Normalizes 'tenant' to 'renter' for backend compatibility
  - Redirects unauthenticated users → `/login`
  - Redirects wrong-role users → their own dashboard
- **Issues**: None

---

### 3.4 Layouts

#### `components/layouts/AdminLayout.jsx` — ~230 lines
- **Purpose**: Admin portal shell (sidebar + topbar)
- **Key Features**:
  - Collapsible sidebar with 4 menu groups: Overview, AI & Data, Management, System
  - System status indicator (green dot + "Online")
  - Dark sidebar (`bg-surface-900 text-white`)
  - Mobile responsive: overlay sidebar with backdrop
  - Sticky top bar with dynamic `title`/`subtitle` props
  - Breadcrumb-style subtitle
- **UI Patterns**: Lucide icons, framer-motion-like CSS transitions, active link detection via `useLocation`
- **Issues**:
  - ⚠️ Some sidebar items link to routes with no corresponding page component (analytics, monitoring, fraud-oversight, feature-toggles, settings)

#### `components/layouts/LandlordLayout.jsx` — ~210 lines
- **Purpose**: Landlord portal shell
- **Key Features**:
  - 14 sidebar menu items with landlord-* color scheme (blue)
  - "Quick Stats" section in sidebar (Listings: 12, Applicants: 28)
  - Same responsive pattern as AdminLayout
- **Issues**:
  - ⚠️ Quick Stats values are **hardcoded** — not fetched from API

#### `components/layouts/TenantLayout.jsx` — ~210 lines
- **Purpose**: Tenant portal shell
- **Key Features**:
  - 11 sidebar menu items with tenant-* color scheme (green/teal)
  - "Trust Score" display in sidebar (92)
  - Same responsive pattern
- **Issues**:
  - ⚠️ Trust Score (92) is **hardcoded** — not dynamically fetched

---

### 3.5 UI Components

#### `components/ui/LoadingSpinner.jsx` — 30 lines
- **Purpose**: Reusable loading indicator
- **Key Features**: Size variants (sm/md/lg), `fullPage` mode (centered in viewport)
- **UI Patterns**: CSS border-spinner with Tailwind animate-spin
- **Issues**: None

#### `components/ui/Logo.jsx` — 42 lines
- **Purpose**: Brand logo with "RentalGuard" text
- **Key Features**: Shield icon, animated green pulse dot, size variants (sm/md/lg)
- **UI Patterns**: Lucide Shield icon, relative positioning for animated dot
- **Issues**: None

#### `components/ui/ThemeToggle.jsx` — 55 lines
- **Purpose**: Light/dark/system theme switcher
- **Key Features**: Two modes — `compact` (icon-only cycle button) and full (3-button selector with labels)
- **UI Patterns**: Uses themeStore, Lucide Sun/Moon/Monitor icons
- **Issues**: None

---

### 3.6 Public Components

#### `components/public/PublicNavbar.jsx` — ~100 lines
- **Purpose**: Fixed top navbar for public pages
- **Key Features**: Scroll-triggered shadow, mobile hamburger menu, Logo component, Sign In / Get Started CTAs, ThemeToggle
- **UI Patterns**: `scroll` event listener, fixed positioning, backdrop blur
- **Issues**: None

#### `components/public/Footer.jsx` — ~110 lines
- **Purpose**: Site footer
- **Key Features**: 4-column grid (Product, Company, Resources, Legal), social links, copyright
- **UI Patterns**: Standard footer grid with Lucide icons
- **Issues**: Links (Privacy, Terms, Contact, etc.) are `href="#"` — non-functional

---

### 3.7 Auth Pages

#### `pages/Login.jsx` — 263 lines
- **Purpose**: Role-aware login page
- **Key Features**:
  - Split-screen layout: left branding panel (gradient by role) + right form
  - Reads `?role=` query param → sets `selectedRole` in authStore
  - Role-specific styling: admin (dark), landlord (blue), tenant (green)
  - Password show/hide toggle
  - Demo credentials card (admin, landlord, tenant)
  - "Remember me" checkbox (non-functional — not persisted)
  - "Forgot password" link (non-functional — `href="#"`)
  - After login: calls `authAPI.login()` then `authAPI.getMe()`, redirects by role
  - Link to Register and GetStarted
- **UI Patterns**: ThemeToggle, toast notifications, btn/input-field CSS classes
- **Issues**:
  - ⚠️ Demo credentials displayed in UI — security concern for production
  - ⚠️ "Remember me" not implemented
  - ⚠️ "Forgot password" not implemented
  - ⚠️ Two API calls for login (login + getMe) — could be combined server-side

#### `pages/Register.jsx` — 282 lines
- **Purpose**: Role-aware registration page
- **Key Features**:
  - Same split-screen layout as Login
  - Reads `?role=` query param (defaults to 'renter')
  - Fields: Full Name, Email, Password
  - Password strength meter (4 levels: Weak/Fair/Good/Strong) with visual bars
  - Terms of Service agreement checkbox (required)
  - Links to Terms/Privacy (non-functional routes)
  - No admin registration option (only landlord/tenant)
  - After register: redirects to `/login?role=<role>`
- **UI Patterns**: Password strength visualization with color-coded bars
- **Issues**:
  - ⚠️ Terms/Privacy links go to `/terms` and `/privacy` — no such routes exist
  - ⚠️ Password `minLength={6}` but placeholder says "Min. 8 characters" — inconsistency

---

### 3.8 Public Pages

#### `pages/public/LandingPage.jsx` — 617 lines
- **Purpose**: Marketing homepage / landing page
- **Key Features**:
  - 8 sections: Hero, Role Selection CTA, Features (6 cards), How It Works (tenant + landlord flow), Security & Trust, AI Technology, Pricing (3 tiers), Testimonials (3 quotes), FAQ (5 items), Final CTA
  - Hero: mock dashboard UI with animated stats, bar chart, risk distribution
  - Pricing: Free ($0), Professional ($29/mo), Enterprise (Custom) — with "Most Popular" badge
  - FAQ: native `<details>` elements for expand/collapse
  - Role selection: Landlord / Tenant cards linking to `/register?role=`
  - Trust bar: "SOC 2 Compliant", "99.9% uptime" etc.
  - Stats: "97.3% Fraud Detection Accuracy", "50K+ Listings Analyzed", "$2.3M Fraud Prevented"
- **UI Patterns**: bg-mesh backgrounds, gradient text, badge-primary, card-hover, shadow-glow, stagger animations
- **Issues**:
  - ⚠️ Marketing claims (SOC 2, 99.9% uptime, 50K+ listings) may not be factual
  - ⚠️ "Contact Sales" and "Talk to Sales" CTAs link to `/contact` — no such route exists
  - ⚠️ Large monolithic component (617 lines) — could be split into section components

#### `pages/public/GetStarted.jsx` — 141 lines
- **Purpose**: Role selection page (choose Landlord or Tenant)
- **Key Features**:
  - Two large cards: "I'm a Landlord" and "I'm a Tenant"
  - Feature lists per role (6 items each)
  - CTAs link to `/register?role=landlord` or `/register?role=renter`
  - Admin access link at bottom → `/login?role=admin`
- **UI Patterns**: hover border color change, scale-110 on icon hover, glow shadows
- **Issues**: None significant

---

### 3.9 Tenant Pages

#### `pages/tenant/Dashboard.jsx` — ~175 lines
- **Purpose**: Tenant home dashboard
- **Key Features**:
  - Stats cards: Total Scans, Safe Listings, Fraud Detected, Scans Left
  - Quick Analyze CTA button
  - Recent Analyses list with risk badges
  - Trust Score display (hardcoded 92)
  - Subscription plan info card
- **UI Patterns**: TenantLayout, stat-card classes, badge-success/warning/danger
- **Issues**:
  - ⚠️ Trust Score (92) is hardcoded — not from API

#### `pages/tenant/Analyze.jsx` — 604 lines
- **Purpose**: Multi-tab fraud analysis tool (core feature)
- **Key Features**:
  - 5 tabs: Text Analysis, URL Analysis, Message Analysis, Conversation Analysis, XAI (Explainability)
  - Text/URL: input with optional price/location, calls `renterAPI.analyzeListing()` or `analyzeUrl()`
  - Message: via FormData upload, calls `renterAPI.analyzeMessage()`
  - Conversation: dynamic message builder (add/remove messages with role), calls `renterAPI.analyzeConversation()`
  - XAI tab: word importance visualization (red=fraud, green=safe tokens), attention weight horizontal bars, counterfactual analysis
  - Extracted `ListingResultCard` subcomponent for consistent result display
- **UI Patterns**: Tab state management, conditional rendering, dynamic form arrays
- **Issues**:
  - ⚠️ Very large monolithic component (604 lines) — should be split per tab
  - ⚠️ No input validation on price field (accepts negative numbers)

#### `pages/tenant/History.jsx` — 438 lines
- **Purpose**: Past analysis history with detail/feedback
- **Key Features**:
  - Stats bar (total, safe, risky, avg score)
  - Filterable/searchable analysis list
  - Detail modal with full analysis results
  - Feedback buttons (Safe / Fraud / Unsure) per analysis
  - Export to HTML or PDF (PDF via base64 decode from API)
  - XAI explanation modal: word importance, attention weights, counterfactuals
- **UI Patterns**: Modal overlays, badge system, base64 download links
- **Issues**: None significant

#### `pages/tenant/Subscription.jsx` — ~300 lines
- **Purpose**: Plan management and payment
- **Key Features**:
  - Current plan display with usage stats
  - Plan cards grid with feature comparison
  - Payment modal with credit card form (card number, expiry, CVV, cardholder name)
  - Payment history table
- **UI Patterns**: Modal with form, card-hover effects
- **Issues**:
  - 🔴 **Critical**: Credit card data handled client-side without PCI compliance — should use Stripe/payment gateway

#### `pages/tenant/Profile.jsx` — ~300 lines
- **Purpose**: User profile management
- **Key Features**:
  - 3 tabs: Profile Info, Security, Notifications
  - Profile editing (name, phone) with save
  - Password change (current + new + confirm) with hide/show toggles
  - Notification toggles (email alerts, analysis reports, security alerts, marketing)
  - Session info display
- **Issues**:
  - ⚠️ Notification preferences are **local state only** — not persisted to API
  - ⚠️ Session info ("Windows • Chrome • Toronto, ON") is **hardcoded**

#### `pages/tenant/ImageVerification.jsx` — ~400 lines
- **Purpose**: Property image fraud detection
- **Key Features**:
  - Two modes: file upload (multi-image with previews) and URL crawl
  - Per-image results: suspicion score, source detection, reverse search matches
  - Loading state with progress indication
- **Issues**:
  - ⚠️ Hardcoded demo/fallback data returned in `catch` blocks

#### `pages/tenant/Payments.jsx` — ~160 lines
- **Purpose**: Payment history viewer
- **Key Features**: Stats cards (Total Paid, This Month, Next Due), filter tabs, payment table
- **Issues**:
  - ⚠️ Hardcoded sample payment data used as fallback

#### `pages/tenant/Listings.jsx` — ~170 lines
- **Purpose**: Browse rental listings
- **Key Features**: Search bar, filters (beds, type, price range slider), listing cards with risk badges, save/apply actions
- **Issues**: None

#### `pages/tenant/AddressCheck.jsx` — ~200 lines
- **Purpose**: Address/location verification
- **Key Features**: Address form, verification results (validity, geocoding, coordinates, confidence score, residential check)
- **Issues**:
  - ⚠️ Province hardcoded to "Ontario"

#### `pages/tenant/SavedListings.jsx` — ~100 lines
- **Purpose**: Bookmarked/saved properties
- **Key Features**: Grid of saved listing cards with remove action
- **Issues**: None

#### `pages/tenant/Applications.jsx` — ~200 lines
- **Purpose**: Application tracking with messaging
- **Key Features**: Application list with status badges, integrated chat panel, message threading
- **Issues**: None

---

### 3.10 Admin Pages (New — Active)

These are the **actively routed** admin pages using `AdminLayout`.

#### `pages/admin/DashboardNew.jsx` — ~200 lines
- **Purpose**: Admin overview dashboard
- **Key Features**: Platform stats cards (users, analyses, fraud detected, models), quick actions grid, feedback summary, auto-learning engine control
- **UI Patterns**: AdminLayout, Link components, toast notifications
- **Issues**: None significant

#### `pages/admin/DatasetsNew.jsx` — ~280 lines
- **Purpose**: Dataset management
- **Key Features**: Search/filter, file upload with format validation, delete confirmation, preview modal with tabular data, analysis modal (overview/quality/recommendations), synthetic dataset generation (configurable samples, fraud ratio, name)
- **UI Patterns**: AdminLayout, modal overlays, file input
- **Issues**: None significant

#### `pages/admin/ModelsNew.jsx` — ~290 lines
- **Purpose**: ML model management and training
- **Key Features**: 3 tabs (Models list, Train Model, BERT Training), model type selection (isolation_forest, random_forest, xgboost, logistic_regression), model cards with activate/deactivate/delete, analysis report modal, BERT training status display with accuracy/F1/precision/recall metrics
- **UI Patterns**: Tab switching, conditional BERT status card
- **Issues**: None significant

#### `pages/admin/UsersNew.jsx` — ~120 lines
- **Purpose**: User management
- **Key Features**: Search + role filter dropdown, user table with role badges (color-coded), deactivate action
- **UI Patterns**: AdminLayout, table-header/table-row CSS classes
- **Issues**: None

#### `pages/admin/AuditLogsNew.jsx` — ~70 lines
- **Purpose**: System activity log viewer
- **Key Features**: Search filter, card-based log display (timestamp, action type, user, IP), simplified compared to old version
- **UI Patterns**: AdminLayout, card list
- **Issues**: Very minimal — no pagination, no expandable details

#### `pages/admin/PlansNew.jsx` — ~200 lines
- **Purpose**: Subscription plan management
- **Key Features**: 8 toggleable features (text_analysis, url_analysis, image_analysis, message_analysis, xai_explanations, export_reports, priority_support, api_access), plan cards with icon mapping (free=Zap, basic=Shield, premium=Crown, enterprise=Rocket), create/edit modal with feature toggle buttons, activate/deactivate, delete
- **UI Patterns**: AdminLayout, icon mapping, modal forms
- **Issues**: None significant

#### `pages/admin/AIEnginesNew.jsx` — ~340 lines
- **Purpose**: AI engine monitoring and testing
- **Key Features**: 6 tabs (Engine Status, Test BERT, Test Message AI, Test Cross-Doc, Training Data, Preprocessing). Engine status grid with active/inactive. BERT test with text input → confidence result. Message analysis test with risk/tactics. Cross-doc test with side-by-side documents. Training data KPIs. Preprocessing pipeline visualization
- **UI Patterns**: AdminLayout, lazy tab loading, grid display
- **Issues**: None significant

---

### 3.11 Admin Pages (Old — Dead Code)

**⚠️ These files exist but are NOT imported in App.jsx. They are dead code.**

| File | Lines | Notes |
|------|-------|-------|
| `admin/Dashboard.jsx` | ~315 | Uses `Layout`, `alert()`, `<a href>` instead of `<Link>`, Recharts BarChart |
| `admin/Datasets.jsx` | 526 | Uses `Layout`, comprehensive analysis report modal |
| `admin/Models.jsx` | 1114 | Uses `Layout`, massive monolith with ROC curves, confusion matrix, A/B testing recs, deployment checklist |
| `admin/Users.jsx` | ~180 | Uses `Layout`, basic user table |
| `admin/AuditLogs.jsx` | ~190 | Uses `Layout`, 11 action type badges |
| `admin/Plans.jsx` | 415 | Uses `Layout`, 7 features (vs 8 in new version) |
| `admin/AIEngines.jsx` | ~460 | Uses `Layout`, Promise.allSettled loading, comprehensive engine status |

---

### 3.12 Landlord Pages (New — Active)

#### `pages/landlord/DashboardNew.jsx` — ~135 lines
- **Purpose**: Landlord home dashboard
- **Key Features**: Welcome banner, 4 stats cards (docs/tenants/images/total verified), quick action cards, recent verifications
- **UI Patterns**: LandlordLayout, badge-success/warning/danger
- **Issues**: None

#### `pages/landlord/DocumentVerificationNew.jsx` — ~190 lines
- **Purpose**: Document fraud detection (upload mode)
- **Key Features**: File upload (JPEG/PNG/GIF/WebP/PDF, max 15MB), 7 document types (paystub, id_card, bank_statement, rental_application, employment_letter, tax_document, utility_bill), risk score bar, summary, risk indicators with severity icons
- **UI Patterns**: LandlordLayout, file preview, gradient risk bar
- **Issues**: None

#### `pages/landlord/TenantVerificationNew.jsx` — ~200 lines
- **Purpose**: Form-based tenant screening
- **Key Features**: Screening form (name, email, phone, income, employment, credit score range, landlord reference), risk score bar, summary, recommendations list
- **UI Patterns**: LandlordLayout, form grid
- **Issues**:
  - ⚠️ Different approach than old version (form-based vs document-based) — inconsistent API contract

#### `pages/landlord/PropertyVerificationNew.jsx` — ~150 lines
- **Purpose**: Property image verification
- **Key Features**: Multi-image upload grid, per-image analysis with preview thumbnails and suspicious/clean badges, overall risk score
- **UI Patterns**: LandlordLayout, image grid
- **Issues**: None

#### `pages/landlord/VerificationHistoryNew.jsx` — ~110 lines
- **Purpose**: Past verification records
- **Key Features**: Fetches up to 100 records, card list with expandable JSON details, action type icons
- **UI Patterns**: LandlordLayout, JSON.stringify display
- **Issues**: No pagination (loads all at once)

#### `pages/landlord/Analytics.jsx` — ~110 lines
- **Purpose**: Property analytics dashboard
- **Key Features**: 5 KPI cards (Total Views, Applications, Revenue, Occupancy Rate, Active Listings), Listing Performance table with conversion rate
- **UI Patterns**: LandlordLayout, stat cards, table
- **Issues**: None

#### `pages/landlord/Applicants.jsx` — ~130 lines
- **Purpose**: Manage tenant applications
- **Key Features**: Status filter tabs (All/Pending/Approved/Viewing/Rejected) with counts, application cards with status badges, actions (Schedule Viewing, Approve, Reject)
- **UI Patterns**: LandlordLayout, filter tabs with active styling
- **Issues**: None

#### `pages/landlord/CreateListing.jsx` — ~170 lines
- **Purpose**: New property listing form
- **Key Features**: Multi-section form: Property Details (title, address, city="Toronto", province="ON", postal code, type), Pricing & Size (rent, beds, baths, sqft), Features (laundry, utilities, date, pet friendly, parking, 9 amenity toggles), Description. Navigates to `/landlord/my-listings` on success
- **UI Patterns**: LandlordLayout, grid layout, toggle buttons for amenities
- **Issues**:
  - ⚠️ City defaults to "Toronto", province to "ON" — Toronto-specific

#### `pages/landlord/LandlordPayments.jsx` — ~110 lines
- **Purpose**: Revenue and payment tracking
- **Key Features**: Revenue stats (Monthly, Active Leases, Annual Projected), income by property list derived from lease data
- **UI Patterns**: LandlordLayout, stat cards
- **Issues**: None

#### `pages/landlord/Leases.jsx` — ~120 lines
- **Purpose**: Lease management
- **Key Features**: Stats (Total/Active Leases, Revenue), lease cards with status badges (active/pending/expired/terminated), tenant name, date range, rent, deposit
- **UI Patterns**: LandlordLayout, status-colored badges
- **Issues**: None

#### `pages/landlord/Messages.jsx` — ~160 lines
- **Purpose**: Tenant messaging system
- **Key Features**: Two-panel UI: contacts sidebar (with unread count badges) + chat area. Message bubbles (blue for sent, gray for received). Send form at bottom
- **UI Patterns**: LandlordLayout, flex layout with overflow-y-auto scroll areas
- **Issues**: None

#### `pages/landlord/MyListings.jsx` — ~160 lines
- **Purpose**: Manage own property listings
- **Key Features**: Stats (Total/Active Listings, Views, Applicants), listing cards with active/inactive badge, views/applicants count, toggle active/delete actions, "New Listing" button
- **UI Patterns**: LandlordLayout, confirm() for delete
- **Issues**:
  - ⚠️ Uses `confirm()` for delete — should use proper modal

#### `pages/landlord/RiskAnalysis.jsx` — ~320 lines
- **Purpose**: Multi-mode risk analysis tool
- **Key Features**: 3 tabs: Risk History (past verifications), Cross-Document Check (multi-file upload with base64 conversion), Full Application Verify (multi-file + optional name). Shared `VerificationResultCard` subcomponent with risk score bar, inconsistencies, flags, documents analyzed, summary
- **UI Patterns**: LandlordLayout, tabs, reusable subcomponent
- **Issues**: None

#### `pages/landlord/Settings.jsx` — ~160 lines
- **Purpose**: Landlord profile & security settings
- **Key Features**: 2 tabs: Profile (avatar placeholder, name editing, email display-only, role/plan info) and Security (password change with current/new/confirm + show/hide toggle)
- **UI Patterns**: LandlordLayout, tabs, form inputs
- **Issues**: None

---

### 3.13 Landlord Pages (Old — Dead Code)

**⚠️ Not imported in App.jsx — dead code.**

| File | Lines | Notes |
|------|-------|-------|
| `landlord/Dashboard.jsx` | ~160 | Uses `Layout` |
| `landlord/DocumentVerification.jsx` | ~310 | Uses `Layout`, supports both upload + base64 modes |
| `landlord/TenantVerification.jsx` | ~330 | Uses `Layout`, **document-based** verification (different from new form-based) |
| `landlord/PropertyVerification.jsx` | ~300 | Uses `Layout`, max 20 images |
| `landlord/VerificationHistory.jsx` | ~250 | Uses `Layout`, paginated (20 per page) |

---

### 3.14 Renter Pages (Legacy Duplicates)

**⚠️ The `pages/renter/` directory contains 4 files that are legacy duplicates. App.jsx imports from `pages/tenant/` instead.**

| File | Lines | Notes |
|------|-------|-------|
| `renter/Dashboard.jsx` | 253 | Uses `Layout`, PieChart (Recharts) — replaced by `tenant/Dashboard.jsx` |
| `renter/Analyze.jsx` | — | Legacy duplicate |
| `renter/History.jsx` | — | Legacy duplicate |
| `renter/Subscription.jsx` | — | Legacy duplicate |

---

### 3.15 CSS

#### `styles/globals.css` — ~280 lines
- **Purpose**: Complete Tailwind design system
- **Key Features**:
  - **Fonts**: Google Fonts import (Inter 300-700, Plus Jakarta Sans 500-800, JetBrains Mono 400-600)
  - **Base layer**: box-border borders, smooth scrolling, font smoothing, body bg/text with dark mode, custom selection color (primary-600), custom scrollbar (::-webkit-scrollbar)
  - **Button system**: `.btn` with sizes (-sm/-md/-lg/-xl) and 7 variants (-primary/-secondary/-outline/-ghost/-danger/-landlord/-tenant). Hover glow effects (`box-shadow`), active scale-down (`transform: scale(0.98)`)
  - **Card system**: `.card` (rounded-2xl, shadow, dark mode), `.card-hover` (lift on hover), `.card-interactive` (cursor pointer), `.card-glass` (backdrop-blur)
  - **Input system**: `.input-field` (rounded-xl, ring focus, dark mode), `.input-label`
  - **Badge system**: `.badge` with 5 variants (-primary/-success/-warning/-danger/-info)
  - **Navigation**: `.nav-link`, `.nav-link-active`
  - **Components**: `.section-heading`, `.stat-card`, `.glass`, `.text-gradient`, `.text-gradient-hero`, `.divider`, `.table-header`, `.table-row`, `.table-cell`
  - **Animations**: `fadeIn` (0.4s), `slideUp` (0.5s), `stagger-1` to `stagger-5` (delayed), `page-enter`
  - **Background**: `.bg-mesh` with multi-radial gradients, light + dark variants
- **Issues**: None — well-structured design system

---

### 3.16 Test Files

#### `__tests__/setup.js` — ~30 lines
- **Purpose**: Vitest test environment setup
- **Key Features**: Imports `@testing-library/jest-dom`, mocks `IntersectionObserver`, `window.matchMedia`, `ResizeObserver`
- **Issues**: None

#### `__tests__/api.test.js` — ~80 lines
- **Purpose**: API service layer tests
- **Key Features**: Mocks axios and authStore. Verifies `adminAPI`, `renterAPI`, `landlordAPI` export expected methods
- **Issues**: Tests only method existence, not behavior

#### `__tests__/App.test.jsx` — ~100 lines
- **Purpose**: Route configuration tests
- **Key Features**: Mocks all page components and guards. Tests route counts
- **Issues**:
  - 🔴 **Route counts are OUTDATED** — test expects 18 routes (2 public + 7 admin + 4 renter + 5 landlord) but App.jsx has 36+ routes. Tests will fail.

#### `__tests__/authStore.test.js` — ~50 lines
- **Purpose**: Auth store unit tests
- **Key Features**: Mocks localStorage, tests initial state (unauthenticated), verifies login/logout function existence
- **Issues**: Tests are very shallow — only check existence, not behavior

---

### 3.17 Routing (App.jsx)

#### `App.jsx` — 139 lines
- **Purpose**: Central routing configuration
- **Total Routes**: 36 + 4 legacy redirects + 1 catch-all

| Segment | Count | Routes |
|---------|-------|--------|
| Public | 4 | `/`, `/get-started`, `/login`, `/register` |
| Tenant | 11 | `/tenant`, `/tenant/listings`, `/tenant/analyze`, `/tenant/verify-images`, `/tenant/verify-address`, `/tenant/saved`, `/tenant/applications`, `/tenant/payments`, `/tenant/history`, `/tenant/subscription`, `/tenant/profile` |
| Landlord | 14 | `/landlord`, `/landlord/listings`, `/landlord/listings/new`, `/landlord/applicants`, `/landlord/risk-analysis`, `/landlord/documents`, `/landlord/tenants`, `/landlord/property-images`, `/landlord/leases`, `/landlord/payments`, `/landlord/messages`, `/landlord/analytics`, `/landlord/history`, `/landlord/settings` |
| Admin | 7 | `/admin`, `/admin/datasets`, `/admin/models`, `/admin/users`, `/admin/audit-logs`, `/admin/plans`, `/admin/ai-engines` |
| Redirects | 4 | `/dashboard`→`/tenant`, `/analyze`→`/tenant/analyze`, `/history`→`/tenant/history`, `/subscription`→`/tenant/subscription` |
| Catch-all | 1 | `*` → `/` |

**Critical insight**: App.jsx imports **only** the "New" page versions. Old page components are never imported or routed.

---

## 4. Design System

### Color Tokens (Tailwind custom)

| Token Family | Usage | Values |
|-------------|-------|--------|
| `surface-*` | Neutral backgrounds/text | 50–950 grayscale |
| `primary-*` | Brand actions/accents | Indigo-like palette |
| `tenant-*` | Tenant portal | Green/teal palette |
| `landlord-*` | Landlord portal | Blue palette |
| `accent-green` | Success states | Green |
| `accent-red` | Error/danger states | Red |
| `accent-amber` | Warning states | Amber |
| `accent-blue` | Info states | Blue |

### Typography

| Class | Font | Usage |
|-------|------|-------|
| body default | Inter | All body text |
| `font-display` | Plus Jakarta Sans | Headings, display text |
| `font-mono` | JetBrains Mono | Code, technical values |
| `text-display-sm/md/lg/xl/2xl` | Custom sizes | Display headings |

### Component Classes

| Class | Description |
|-------|-------------|
| `.btn` + variants | Full button system with 7 color variants, 4 sizes |
| `.card` + variants | Card containers (default, hover, interactive, glass) |
| `.input-field` / `.input-label` | Form inputs with dark mode |
| `.badge` + variants | Status labels (5 color variants) |
| `.nav-link` / `.nav-link-active` | Navigation items |
| `.stat-card` | Statistics display cards |
| `.glass` | Glassmorphism effect |
| `.text-gradient` / `.text-gradient-hero` | Gradient text effects |
| `.bg-mesh` | Multi-radial gradient backgrounds |

---

## 5. Issues & Recommendations

### 🔴 Critical

| # | Issue | Location | Recommendation |
|---|-------|----------|----------------|
| 1 | Credit card data handled client-side without PCI compliance | `tenant/Subscription.jsx` | Integrate Stripe Elements or similar payment gateway |
| 2 | Demo credentials exposed in login UI | `Login.jsx` line ~240 | Remove or gate behind `NODE_ENV=development` |
| 3 | Test suite outdated — route counts wrong | `App.test.jsx` | Update test expectations to match 36 routes |

### ⚠️ Important

| # | Issue | Location | Recommendation |
|---|-------|----------|----------------|
| 4 | Hardcoded API URL | `api.js` | Use `import.meta.env.VITE_API_URL` |
| 5 | Hardcoded Trust Score (92) | `TenantLayout.jsx`, `tenant/Dashboard.jsx` | Fetch from API via `renterAPI.getStats()` |
| 6 | Hardcoded Quick Stats (12/28) | `LandlordLayout.jsx` | Fetch from API via `landlordAPI.getDashboardStats()` |
| 7 | Province hardcoded to Ontario | `tenant/AddressCheck.jsx` | Make configurable or auto-detect |
| 8 | City defaults to Toronto | `landlord/CreateListing.jsx` | Allow user selection |
| 9 | Hardcoded fallback data in catch blocks | `tenant/ImageVerification.jsx`, `tenant/Payments.jsx` | Return proper error state instead |
| 10 | Notifications not persisted | `tenant/Profile.jsx` | Add API endpoint + persist |
| 11 | Session info hardcoded | `tenant/Profile.jsx` | Fetch real session data or remove |
| 12 | No token refresh / expiry handling | `authStore.js` | Implement refresh token rotation |
| 13 | Duplicate API aliases | `api.js` | Remove `triggerAutoLearn` alias |
| 14 | `confirm()` used for destructive actions | `landlord/MyListings.jsx`, admin pages | Replace with confirmation modal |
| 15 | No accessibility (aria-labels) | Throughout | Add ARIA attributes to interactive elements |
| 16 | Missing routes in sidebar | `AdminLayout.jsx` | Add pages for analytics, monitoring, settings, etc. or remove menu items |
| 17 | "Forgot password" link non-functional | `Login.jsx` | Implement or remove |
| 18 | Terms/Privacy routes don't exist | `Register.jsx` | Create pages or link externally |
| 19 | Marketing claims may be unverified | `LandingPage.jsx` | Verify or mark as aspirational |

### 💡 Code Quality

| # | Issue | Location | Recommendation |
|---|-------|----------|----------------|
| 20 | 1114-line monolith | `admin/Models.jsx` (dead code) | N/A — dead code, can delete |
| 21 | 617-line monolith | `public/LandingPage.jsx` | Split into section components |
| 22 | 604-line monolith | `tenant/Analyze.jsx` | Split each tab into its own component |
| 23 | Tests are shallow | All test files | Add behavioral tests, integration tests |
| 24 | ~2,500 lines of dead code | Old admin + landlord + renter pages | Delete or archive |

---

## 6. Dead Code Inventory

Files that exist in the codebase but are **never imported or routed** in App.jsx:

| File | Lines | Can Delete? |
|------|-------|-------------|
| `admin/Dashboard.jsx` | ~315 | ✅ Yes |
| `admin/Datasets.jsx` | 526 | ✅ Yes |
| `admin/Models.jsx` | 1114 | ✅ Yes |
| `admin/Users.jsx` | ~180 | ✅ Yes |
| `admin/AuditLogs.jsx` | ~190 | ✅ Yes |
| `admin/Plans.jsx` | 415 | ✅ Yes |
| `admin/AIEngines.jsx` | ~460 | ✅ Yes |
| `landlord/Dashboard.jsx` | ~160 | ✅ Yes |
| `landlord/DocumentVerification.jsx` | ~310 | ✅ Yes |
| `landlord/TenantVerification.jsx` | ~330 | ✅ Yes |
| `landlord/PropertyVerification.jsx` | ~300 | ✅ Yes |
| `landlord/VerificationHistory.jsx` | ~250 | ✅ Yes |
| `renter/Dashboard.jsx` | 253 | ✅ Yes |
| `renter/Analyze.jsx` | ~??? | ✅ Yes |
| `renter/History.jsx` | ~??? | ✅ Yes |
| `renter/Subscription.jsx` | ~??? | ✅ Yes |
| `components/Layout.jsx` | ~??? | ✅ Yes |
| **Total estimated** | **~4,800+** | |

---

## 7. Summary Statistics

| Metric | Value |
|--------|-------|
| Total source files | 65+ |
| Active (routed) pages | 36 |
| Dead code files | 17 |
| Estimated total LOC | ~12,500 |
| Estimated dead LOC | ~4,800 |
| Estimated active LOC | ~7,700 |
| API endpoints consumed | 70+ |
| Zustand stores | 2 |
| Layout components | 3 (active) + 1 (legacy) |
| Reusable UI components | 5 |
| Test files | 4 |
| Test coverage | Low (method existence only) |
| Dark mode support | ✅ Full |
| Mobile responsive | ✅ All layouts |
| User roles | 3 (admin, landlord, tenant/renter) |

---

*End of report.*
