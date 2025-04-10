import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import {
  Download,
  Users,
  MessageSquare,
  Video,
  Mic,
  Filter,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import {
  departments,
  mockFeedbackData,
  monthlySentimentData,
  feedbackTopicsData,
  departmentPerformanceData,
  COLORS,
} from "@/data/constants";
import Cookies from "js-cookie";
import axios from "axios";
import { set } from "date-fns";

const AdminDashboard = () => {
  const navigate = useNavigate();
  const { toast } = useToast();

  const [adminInfo, setAdminInfo] = useState<any>(null);
  const [selectedDepartment, setSelectedDepartment] = useState("all");
  const [feedbackList, setFeedbackList] = useState<any[]>([]);
  const [allFeedbacks, setAllFeedbacks] = useState<any[]>([]);
  const [activeResponseIndex, setActiveResponseIndex] = useState(null);
  const [responseMessage, setResponseMessage] = useState("");

  useEffect(() => {
    const fetchAllFeedbacks = async () => {
      try {
        const res = await axios.get(
          "https://careloop.onrender.com/bloom/v1/api/admin/fetchAll", //http://localhost:3000   //https://careloop.onrender.com
          {
            withCredentials: true,
          }
        );
        const data = res.data.allFeedbacks;
        setAllFeedbacks(data);
        console.log("Fetched feedbacks:", data);
      } catch (error) {
        console.error("Error fetching feedbacks:", error.message);
      }
    };

    const adminData = Cookies.get("adminLoginData");
    console.log("Admin data from cookie:", adminData);

    if (adminData) {
      const parsedData = JSON.parse(adminData);
      setAdminInfo(parsedData);
      fetchAllFeedbacks();
    } else {
      console.warn("No admin cookie found. Redirecting to login.");
      navigate("/login/admin");
    }
  }, [navigate]);

  // Separate effect for filtering feedbacks based on selectedDepartment and allFeedbacks
  useEffect(() => {
    if (selectedDepartment === "all") {
      setFeedbackList(allFeedbacks);
    } else {
      setFeedbackList(
        allFeedbacks.filter(
          (item) =>
            item.departmentId?.toLowerCase().trim() ===
            selectedDepartment.toLowerCase().trim()
        )
      );
    }
  }, [selectedDepartment, allFeedbacks]);

  const handleLogout = async () => {
    try {
      const response = await fetch(
        "https://careloop.onrender.com/bloom/v1/api/admin/out",
        {
          method: "POST",
          credentials: "include",
        }
      );

      const data = await response.json();

      console.log("Logout response:", data);

      // Clear sessionStorage
      sessionStorage.removeItem("adminUser");

      // Remove cookie (in case backend didn't or for double-safety)
      Cookies.remove("adminLoginData");

      navigate("/");
    } catch (error) {
      console.error("Logout error:", error.message);
    }
  };

  const handleDepartmentChange = (value: string) => {
    setSelectedDepartment(value);
    console.log("Selected department:", value);
  };

  const handleDownloadActionPlan = () => {
    // In a real application, this would generate and download a report
    toast({
      title: "Action Plan Generated",
      description: "Weekly action plan has been downloaded successfully.",
    });
  };

  const getSentimentBadge = (sentiment: string) => {
    switch (sentiment) {
      case "positive":
        return <Badge className="bg-green-500">Positive</Badge>;
      case "neutral":
        return <Badge className="bg-blue-500">Neutral</Badge>;
      case "negative":
        return <Badge className="bg-red-500">Negative</Badge>;
      default:
        return null;
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case "text":
        return <MessageSquare className="h-4 w-4 text-gray-500" />;
      case "voice":
        return <Mic className="h-4 w-4 text-gray-500" />;
      case "video":
        return <Video className="h-4 w-4 text-gray-500" />;
      default:
        return null;
    }
  };

  const handleRespondClick = (index) => {
    setActiveResponseIndex(activeResponseIndex === index ? null : index);
    setResponseMessage("");
  };

  const handleSubmitResponse = (idx) => {
    // Replace this with actual API logic to save response
    console.log("Submitting response for index", idx, ":", responseMessage);
    setActiveResponseIndex(null); // Hide the input box
  };

  const filteredFeedbacks = feedbackList.filter(
    (f) => f.hospitalID === adminInfo.hospitalId
  );

  if (!adminInfo) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        Loading...
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 shadow-sm">
        <div className="container-custom py-4">
          <div className="flex justify-between items-center">
            <div className="flex items-center space-x-4">
              <div className="w-10 h-10 bg-primary rounded-full flex items-center justify-center text-white font-bold">
                A
              </div>
              <div>
                <h1 className="font-medium">
                  {adminInfo.name || "Admin User"}
                </h1>
                <p className="text-sm text-gray-600">{adminInfo.email}</p>
              </div>
            </div>
            <Button variant="outline" onClick={handleLogout}>
              Logout
            </Button>
          </div>
        </div>
      </header>

      <main className="container-custom py-8">
        <div className="mb-8 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div>
            <h1 className="text-2xl font-bold">Admin Dashboard</h1>
            <p className="text-gray-600">Manage and analyze patient feedback</p>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center space-x-2">
              <Filter className="h-4 w-4 text-gray-500" />
              <Select
                value={selectedDepartment}
                onValueChange={handleDepartmentChange}
              >
                <SelectTrigger className="w-[200px]">
                  <SelectValue placeholder="All Departments" />
                </SelectTrigger>
                <SelectContent>
                  {departments.map((dept) => (
                    <SelectItem key={dept.id} value={dept.id}>
                      {dept.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <Button onClick={handleDownloadActionPlan} className="space-x-2">
              <Download className="h-4 w-4" />
              <span>Download Action Plan</span>
            </Button>
          </div>
        </div>

        {/* Dashboard Overview Cards */}
        <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <Card>
            <CardContent className="p-6">
              <div className="flex items-center space-x-4">
                <div className="bg-blue-100 p-3 rounded-full">
                  <Users className="h-6 w-6 text-primary" />
                </div>
                <div>
                  <p className="text-gray-600 text-sm">Total Feedback</p>
                  <h3 className="text-2xl font-bold">{filteredFeedbacks.length}</h3>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center space-x-4">
                <div className="bg-green-100 p-3 rounded-full">
                  <MessageSquare className="h-6 w-6 text-green-600" />
                </div>
                <div>
                  <p className="text-gray-600 text-sm">Positive</p>
                  <h3 className="text-2xl font-bold">
                    {
                      filteredFeedbacks.filter((f) => f.sentimentIndex === 1)
                        .length
                    }
                  </h3>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center space-x-4">
                <div className="bg-yellow-100 p-3 rounded-full">
                  <MessageSquare className="h-6 w-6 text-yellow-600" />
                </div>
                <div>
                  <p className="text-gray-600 text-sm">Neutral</p>
                  <h3 className="text-2xl font-bold">
                    {
                      filteredFeedbacks.filter((f) => f.sentimentIndex === 0)
                        .length
                    }
                  </h3>
                </div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardContent className="p-6">
              <div className="flex items-center space-x-4">
                <div className="bg-red-100 p-3 rounded-full">
                  <MessageSquare className="h-6 w-6 text-red-600" />
                </div>
                <div>
                  <p className="text-gray-600 text-sm">Negative</p>
                  <h3 className="text-2xl font-bold">
                    {
                      filteredFeedbacks.filter((f) => f.sentimentIndex === -1)
                        .length
                    }
                  </h3>
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Main Content Tabs */}
        <Tabs defaultValue="feedback">
          <TabsList className="mb-6">
            <TabsTrigger value="feedback">Feedback List</TabsTrigger>
            <TabsTrigger value="analytics">Analytics</TabsTrigger>
          </TabsList>

          <TabsContent value="feedback">
            <Card>
              <CardHeader>
                <CardTitle>Patient Feedback</CardTitle>
                <CardDescription>
                  Review and manage feedback from patients
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto">
                  {feedbackList.length > 0 && (
                    <div className="mt-12 max-w-5xl mx-auto space-y-6 px-4 sm:px-6 lg:px-0">
                      {feedbackList
                      .filter((fb)=>fb.hospitalID === adminInfo.hospitalId)
                      .map((fb, idx) => (
                        <Card
                          key={idx}
                          className="bg-white shadow-md border border-gray-200 rounded-2xl hover:shadow-lg transition-shadow duration-300"
                        >
                          <CardHeader className="pb-3 border-b border-gray-100">
                            <CardTitle className="text-lg font-semibold text-gray-900">
                              Department: {fb.departmentId}
                            </CardTitle>
                            <CardDescription className="text-sm text-gray-600">
                              Topic: {fb.topic}
                            </CardDescription>
                          </CardHeader>
                          <CardContent className="space-y-3 text-sm text-gray-700 p-4">
                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                              <p>
                                <span className="font-medium text-gray-900">
                                  Patient ID:
                                </span>{" "}
                                {fb.patientID}
                              </p>
                              <p>
                                <span className="font-medium text-gray-900">
                                  Hospital ID:
                                </span>{" "}
                                {fb.hospitalID}
                              </p>
                              <p>
                                <span className="font-medium text-gray-900">
                                  Sentiment:
                                </span>{" "}
                                <span
                                  className={`${
                                    fb.sentimentIndex === -1
                                      ? "text-red-500"
                                      : fb.sentimentIndex === 1
                                      ? "text-green-600"
                                      : "text-yellow-600"
                                  } font-semibold`}
                                >
                                  {fb.sentimentIndex === -1
                                    ? "Negative"
                                    : fb.sentimentIndex === 1
                                    ? "Positive"
                                    : "Neutral"}
                                </span>
                              </p>
                              <p>
                                <span className="font-medium text-gray-900">
                                  Content Type:
                                </span>{" "}
                                {fb.contentTypeIndex === 0
                                  ? "Text"
                                  : fb.contentTypeIndex === 1
                                  ? "Voice"
                                  : "Video"}
                              </p>
                            </div>

                            <div>
                              <p>
                                <span className="font-medium text-gray-900">
                                  Feedback:
                                </span>{" "}
                                {fb.textContent || "—"}
                              </p>
                            </div>

                            {fb.mediaContent && (
                              <p>
                                <a
                                  href={fb.mediaContent}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="text-blue-600 hover:underline"
                                >
                                  View Media
                                </a>
                              </p>
                            )}

                            <div className="pt-2 border-t border-gray-100 flex flex-col gap-5">
                              <div className="flex gap-10 items-center">
                                <p>
                                  <span className="font-medium text-gray-900">
                                    Admin Response:
                                  </span>{" "}
                                  {fb.response_status
                                    ? fb.response
                                    : "Not yet responded"}
                                </p>
                                <Button
                                  variant="outline"
                                  onClick={() => handleRespondClick(idx)}
                                >
                                  Respond
                                </Button>
                              </div>
                              {activeResponseIndex === idx && (
                                <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
                                  <input
                                    type="text"
                                    value={responseMessage}
                                    onChange={(e) =>
                                      setResponseMessage(e.target.value)
                                    }
                                    placeholder="Enter your response..."
                                    className="border rounded-md px-4 py-2 w-full sm:w-2/3"
                                  />
                                  <Button
                                    onClick={() => handleSubmitResponse(idx)}
                                  >
                                    Submit
                                  </Button>

                                  <Button
                                    onClick={() => handleRespondClick(idx)}
                                    variant="outline"
                                  >
                                    Close
                                  </Button>
                                </div>
                              )}
                            </div>
                          </CardContent>
                        </Card>
                      ))}
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="analytics">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <Card>
                <CardHeader>
                  <CardTitle>Monthly Sentiment Analysis</CardTitle>
                  <CardDescription>
                    Comparison of positive, negative, and neutral feedback over
                    time
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-[400px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={monthlySentimentData}
                        margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="month" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Bar
                          dataKey="positive"
                          fill="#22c55e"
                          name="Positive"
                        />
                        <Bar dataKey="neutral" fill="#3b82f6" name="Neutral" />
                        <Bar
                          dataKey="negative"
                          fill="#ef4444"
                          name="Negative"
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Feedback Topics</CardTitle>
                  <CardDescription>
                    Distribution of feedback by topic categories
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-[400px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={feedbackTopicsData}
                          cx="50%"
                          cy="50%"
                          labelLine={true}
                          outerRadius={150}
                          fill="#8884d8"
                          dataKey="value"
                          label={({ name, percent }) =>
                            `${name} ${(percent * 100).toFixed(0)}%`
                          }
                        >
                          {feedbackTopicsData.map((entry, index) => (
                            <Cell
                              key={`cell-${index}`}
                              fill={COLORS[index % COLORS.length]}
                            />
                          ))}
                        </Pie>
                        <Tooltip formatter={(value) => `${value} feedbacks`} />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              <Card className="lg:col-span-2">
                <CardHeader>
                  <CardTitle>Department Performance</CardTitle>
                  <CardDescription>
                    Comparison of sentiment across different departments
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="h-[400px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={departmentPerformanceData}
                        margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                        layout="vertical"
                      >
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis type="number" />
                        <YAxis dataKey="department" type="category" />
                        <Tooltip />
                        <Legend />
                        <Bar
                          dataKey="positive"
                          stackId="a"
                          fill="#22c55e"
                          name="Positive"
                        />
                        <Bar
                          dataKey="neutral"
                          stackId="a"
                          fill="#3b82f6"
                          name="Neutral"
                        />
                        <Bar
                          dataKey="negative"
                          stackId="a"
                          fill="#ef4444"
                          name="Negative"
                        />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
};

export default AdminDashboard;
