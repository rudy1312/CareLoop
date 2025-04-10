import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { useToast } from "@/hooks/use-toast";
import { z } from "zod";
import axios from "axios";

const patientSchema = z.object({
  name: z.string().min(3, "Name must be at least 3 characters"),
  patientId: z.string().min(3, "ID must be at least 3 characters"),
  mobileNumber: z.string().min(10, "Mobile number must be at least 10 digits"),
  hospitalId: z.string().min(3, "Hospital ID must be at least 3 characters"),
});

type PatientForm = z.infer<typeof patientSchema>;

const PatientRegister = () => {
  const navigate = useNavigate();
  const { toast } = useToast();

  const [formData, setFormData] = useState<PatientForm>({
    name: "",
    patientId: "",
    mobileNumber: "",
    hospitalId: "",
  });

  const [errors, setErrors] = useState<
    Partial<Record<keyof PatientForm, string>>
  >({});
  const [isLoading, setIsLoading] = useState(false);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
    // Clear error when user types
    if (errors[name as keyof PatientForm]) {
      setErrors((prev) => ({ ...prev, [name]: undefined }));
    }
  };

  const validateForm = () => {
    try {
      patientSchema.parse(formData);
      return true;
    } catch (error) {
      if (error instanceof z.ZodError) {
        const newErrors: Partial<Record<keyof PatientForm, string>> = {};
        error.errors.forEach((err) => {
          if (err.path[0]) {
            newErrors[err.path[0] as keyof PatientForm] = err.message;
          }
        });
        setErrors(newErrors);
      }
      return false;
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!validateForm()) return;

    setIsLoading(true);

    // Simulate registration process
    try {
      const response = await axios.post(
        "https://careloop.onrender.com/bloom/v1/api/patient/reg",
        {
          fullName: formData.name,
          patientID: formData.patientId,
          mobile: formData.mobileNumber,
          hospitalID: formData.hospitalId,
        }
      );

      console.log("Backend response:", response.data);

      toast({
        title: "Registration successful",
        description: "Your account has been created. You can now log in.",
      });

      navigate("/login/patient");
    } catch (error: any) {
      console.error("Registration error:", error);

      toast({
        title: "Registration failed",
        description:
          error?.response?.data?.message ||
          "Something went wrong. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-cyan-50 p-4">
      <div className="w-full max-w-md">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold">Patient Registration</h1>
          <p className="text-gray-600 mt-2">
            Create your account to provide feedback
          </p>
        </div>

        <Card className="glass-card border-gray-200">
          <CardHeader>
            <CardTitle className="text-xl">Patient Sign Up</CardTitle>
            <CardDescription>
              Please fill in your details to create an account
            </CardDescription>
          </CardHeader>
          <form onSubmit={handleSubmit}>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="name">Full Name</Label>
                <Input
                  id="name"
                  name="name"
                  placeholder="John Doe"
                  value={formData.name}
                  onChange={handleChange}
                  className={errors.name ? "border-red-500" : ""}
                />
                {errors.name && (
                  <p className="text-red-500 text-xs mt-1">{errors.name}</p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="patientId">Patient ID</Label>
                <Input
                  id="patientId"
                  name="patientId"
                  placeholder="PAT123456"
                  value={formData.patientId}
                  onChange={handleChange}
                  className={errors.patientId ? "border-red-500" : ""}
                />
                {errors.patientId && (
                  <p className="text-red-500 text-xs mt-1">
                    {errors.patientId}
                  </p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="mobileNumber">Mobile Number</Label>
                <Input
                  id="mobileNumber"
                  name="mobileNumber"
                  type="tel"
                  placeholder="(555) 123-4567"
                  value={formData.mobileNumber}
                  onChange={handleChange}
                  className={errors.mobileNumber ? "border-red-500" : ""}
                />
                {errors.mobileNumber && (
                  <p className="text-red-500 text-xs mt-1">
                    {errors.mobileNumber}
                  </p>
                )}
              </div>

              <div className="space-y-2">
                <Label htmlFor="hospitalId">Hospital ID</Label>
                <Input
                  id="hospitalId"
                  name="hospitalId"
                  placeholder="HOSP001"
                  value={formData.hospitalId}
                  onChange={handleChange}
                  className={errors.hospitalId ? "border-red-500" : ""}
                />
                {errors.hospitalId && (
                  <p className="text-red-500 text-xs mt-1">
                    {errors.hospitalId}
                  </p>
                )}
              </div>
            </CardContent>

            <CardFooter className="flex-col space-y-4">
              <Button
                type="submit"
                className="w-full btn-hover"
                disabled={isLoading}
              >
                {isLoading ? "Creating Account..." : "Register"}
              </Button>

              <p className="text-sm text-center">
                Already have an account?{" "}
                <Link
                  to="/login/patient"
                  className="text-primary font-medium hover:underline"
                >
                  Log in
                </Link>
              </p>
            </CardFooter>
          </form>
        </Card>
      </div>
    </div>
  );
};

export default PatientRegister;
