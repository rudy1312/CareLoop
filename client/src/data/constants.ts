// Patient and Admin Dashboard Constants

// Departments list for both dashboards
export const departments = [
  { id: "all", name: "All Departments" },
  { id: "emergency", name: "Emergency" },
  { id: "cardiology", name: "Cardiology" },
  { id: "pediatrics", name: "Pediatrics" },
  { id: "orthopedics", name: "Orthopedics" },
  { id: "neurology", name: "Neurology" },
  { id: "oncology", name: "Oncology" },
  { id: "radiology", name: "Radiology" },
  { id: "pharmacy", name: "Pharmacy" },
];

// Admin Dashboard Constants

// Mock feedback data for admin dashboard
export const mockFeedbackData = [
  {
    id: 1,
    patientName: "John Smith",
    patientId: "PAT78945",
    department: "emergency",
    departmentName: "Emergency",
    type: "text",
    content:
      "The wait time was too long, but the staff was very helpful once I was seen.",
    sentiment: "neutral",
    date: "2023-04-01",
  },
  {
    id: 2,
    patientName: "Emily Johnson",
    patientId: "PAT56123",
    department: "cardiology",
    departmentName: "Cardiology",
    type: "voice",
    content: "voice_recording.mp3",
    sentiment: "positive",
    date: "2023-04-02",
  },
  {
    id: 3,
    patientName: "Michael Brown",
    patientId: "PAT12378",
    department: "pediatrics",
    departmentName: "Pediatrics",
    type: "video",
    content: "video_feedback.mp4",
    sentiment: "positive",
    date: "2023-04-03",
  },
  {
    id: 4,
    patientName: "Sarah Davis",
    patientId: "PAT90567",
    department: "orthopedics",
    departmentName: "Orthopedics",
    type: "text",
    content:
      "The doctor explained my condition very clearly and the treatment plan is working well.",
    sentiment: "positive",
    date: "2023-04-04",
  },
  {
    id: 5,
    patientName: "Robert Wilson",
    patientId: "PAT34521",
    department: "neurology",
    departmentName: "Neurology",
    type: "text",
    content:
      "The billing department made several errors on my statement and it took multiple calls to resolve.",
    sentiment: "negative",
    date: "2023-04-05",
  },
  {
    id: 6,
    patientName: "Jennifer Lee",
    patientId: "PAT67890",
    department: "oncology",
    departmentName: "Oncology",
    type: "voice",
    content: "voice_feedback.mp3",
    sentiment: "positive",
    date: "2023-04-06",
  },
  {
    id: 7,
    patientName: "David Miller",
    patientId: "PAT23456",
    department: "radiology",
    departmentName: "Radiology",
    type: "text",
    content: "The facility was very clean and the staff was professional.",
    sentiment: "positive",
    date: "2023-04-07",
  },
  {
    id: 8,
    patientName: "Lisa Taylor",
    patientId: "PAT78901",
    department: "pharmacy",
    departmentName: "Pharmacy",
    type: "text",
    content: "Long wait times and the pharmacist seemed rushed.",
    sentiment: "negative",
    date: "2023-04-08",
  },
];

// Monthly sentiment data for bar chart
export const monthlySentimentData = [
  { month: "Jan", positive: 42, neutral: 28, negative: 14 },
  { month: "Feb", positive: 38, neutral: 30, negative: 18 },
  { month: "Mar", positive: 49, neutral: 27, negative: 12 },
  { month: "Apr", positive: 55, neutral: 29, negative: 10 },
  { month: "May", positive: 47, neutral: 31, negative: 16 },
  { month: "Jun", positive: 52, neutral: 25, negative: 13 },
];

// Feedback topics data for pie chart
export const feedbackTopicsData = [
  { name: "Staff", value: 35 },
  { name: "Cleanliness", value: 20 },
  { name: "Wait Time", value: 18 },
  { name: "Communication", value: 15 },
  { name: "Billing", value: 12 },
];

// Department performance data for bar chart
export const departmentPerformanceData = [
  { department: "Emergency", positive: 65, neutral: 20, negative: 15 },
  { department: "Cardiology", positive: 75, neutral: 15, negative: 10 },
  { department: "Pediatrics", positive: 80, neutral: 15, negative: 5 },
  { department: "Orthopedics", positive: 70, neutral: 20, negative: 10 },
  { department: "Neurology", positive: 60, neutral: 25, negative: 15 },
  { department: "Oncology", positive: 65, neutral: 25, negative: 10 },
  { department: "Radiology", positive: 75, neutral: 15, negative: 10 },
  { department: "Pharmacy", positive: 55, neutral: 25, negative: 20 },
];

// Chart colors
export const COLORS = ["#0088FE", "#00C49F", "#FFBB28", "#FF8042", "#8884D8"];

// Pricing page constants
export const pricingPlans = [
  {
    name: "Starter",
    description: "Perfect for small clinics",
    monthlyPrice: 149,
    annualPrice: 129,
    features: [
      "Up to 5 departments",
      "Text feedback collection",
      "Basic sentiment analysis",
      "Weekly reports",
      "1 admin user",
      "Email support",
    ],
  },
  {
    name: "Professional",
    description: "Ideal for mid-sized hospitals",
    monthlyPrice: 299,
    annualPrice: 249,
    popular: true,
    features: [
      "Up to 15 departments",
      "Text, voice, and video feedback",
      "Advanced sentiment analysis",
      "Department-specific dashboards",
      "5 admin users",
      "Weekly action plans",
      "Priority email support",
    ],
  },
  {
    name: "Enterprise",
    description: "For large healthcare systems",
    monthlyPrice: 599,
    annualPrice: 499,
    features: [
      "Unlimited departments",
      "All feedback formats",
      "Advanced analytics and trends",
      "Custom reporting",
      "Unlimited admin users",
      "API access",
      "Dedicated account manager",
      "24/7 phone support",
    ],
  },
];

// FAQ data for pricing page
export const faqData = [
  {
    question: "Can I upgrade or downgrade my plan later?",
    answer:
      "Yes, you can change your plan at any time. If you upgrade, we'll prorate the difference. If you downgrade, the new rate will apply at the start of your next billing cycle.",
  },
  {
    question: "Is there a contract or commitment?",
    answer:
      "No long-term contracts. You can cancel your subscription at any time, though we do not offer refunds for partial months.",
  },
  {
    question: "Do you offer a free trial?",
    answer:
      "Yes, we offer a 14-day free trial on all plans. No credit card required to start your trial.",
  },
  {
    question: "How secure is patient feedback data?",
    answer:
      "CareLoop is HIPAA compliant and uses industry-leading encryption to protect all patient data. We never share or sell your data.",
  },
  {
    question: "Can I customize the feedback forms?",
    answer:
      "Yes, all plans allow for customization of feedback forms to match your hospital's branding and specific department needs.",
  },
  {
    question: "Do you offer any discounts for non-profits?",
    answer:
      "Yes, we offer special pricing for non-profit healthcare organizations. Please contact our sales team for details.",
  },
];
