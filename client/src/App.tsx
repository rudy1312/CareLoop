import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, useLocation } from "react-router-dom";
import Navbar from "./components/Navbar";
import Footer from "./components/Footer";
import ScrollToTop from "./components/ScrollToTop";
import Index from "./pages/Index";
import AboutPage from "./pages/AboutPage";
import PricingPage from "./pages/PricingPage";
import PatientLogin from "./pages/PatientLogin";
import PatientRegister from "./pages/PatientRegister";
import AdminLogin from "./pages/AdminLogin";
import AdminRegister from "./pages/AdminRegister";
import PatientDashboard from "./pages/PatientDashboard";
import AdminDashboard from "./pages/AdminDashboard";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

// Create a layout component to handle conditional rendering of Navbar and Footer
const Layout = ({ children }: { children: React.ReactNode }) => {
  const location = useLocation();
  
  // Function to determine if navbar and footer should be shown
  const shouldShowNavbarFooter = (pathname: string) => {
    // Don't show navbar/footer on dashboard and login pages
    return !pathname.includes('/dashboard') && !pathname.includes('/login') && !pathname.includes('/register');
  };

  return (
    <>
      {shouldShowNavbarFooter(location.pathname) && <Navbar />}
      {children}
      {shouldShowNavbarFooter(location.pathname) && <Footer />}
    </>
  );
};

const App = () => {
  return (
    <QueryClientProvider client={queryClient}>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <ScrollToTop />
          <Layout>
            <Routes>
              <Route path="/" element={<Index />} />
              <Route path="/about" element={<AboutPage />} />
              <Route path="/pricing" element={<PricingPage />} />
              <Route path="/login/patient" element={<PatientLogin />} />
              <Route path="/register/patient" element={<PatientRegister />} />
              <Route path="/login/admin" element={<AdminLogin />} />
              <Route path="/register/admin" element={<AdminRegister />} />
              <Route path="/patient/dashboard" element={<PatientDashboard />} />
              <Route path="/admin/dashboard" element={<AdminDashboard />} />
              {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
              <Route path="*" element={<NotFound />} />
            </Routes>
          </Layout>
        </BrowserRouter>
      </TooltipProvider>
    </QueryClientProvider>
  );
};

export default App;
