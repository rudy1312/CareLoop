import { ArrowRight, BarChart, MessageCircle, Video, Zap } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";

const Index = () => {
  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="bg-gradient-to-br from-blue-50 to-cyan-50 section-padding px-20">
        <div className="container-custom">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            <div className="space-y-6 max-w-xl">
              <h1 className="heading-1 text-gray-900">
                Capture and analyze patient feedback with ease
              </h1>
              <p className="text-lg text-gray-600">
                CareLoop helps hospitals improve patient satisfaction by
                collecting and analyzing feedback in multiple formats - text,
                voice, and video.
              </p>
              <div className="flex flex-wrap gap-4 pt-4">
                <Link to="/login/admin">
                  <Button size="lg" className="btn-hover rounded-full px-6">
                    Get Started
                    <ArrowRight className="ml-2 h-4 w-4" />
                  </Button>
                </Link>
                <Link to="/demo">
                  <Button
                    variant="outline"
                    size="lg"
                    className="btn-hover rounded-full px-6"
                  >
                    Request Demo
                  </Button>
                </Link>
              </div>
            </div>
            <div className="relative">
              <div className="glass-card rounded-2xl p-6 shadow-xl">
                <img
                  src="https://images.unsplash.com/photo-1581594549595-35f6edc7b762?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80"
                  alt="Dashboard Preview"
                  className="rounded-lg shadow-sm w-full"
                />
              </div>
              <div className="absolute -bottom-6 -left-6 bg-white p-4 rounded-xl shadow-lg animate-fade-in">
                <div className="flex items-center gap-3">
                  <div className="bg-green-100 p-2 rounded-full">
                    <Zap className="h-5 w-5 text-green-600" />
                  </div>
                  <div>
                    <p className="text-sm font-medium">Weekly insights</p>
                    <p className="text-xs text-gray-500">
                      Feedback improved by 27%
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="section-padding px-20">
        <div className="container-custom">
          <div className="text-center max-w-3xl mx-auto mb-16">
            <h2 className="heading-2 text-gray-900 mb-4">
              A complete solution for patient feedback
            </h2>
            <p className="text-lg text-gray-600">
              Collect, analyze, and take action on patient feedback all in one
              platform
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {[
              {
                icon: <MessageCircle className="h-6 w-6 text-primary" />,
                title: "Multi-format Feedback",
                description:
                  "Collect patient feedback in text, voice, or video format for comprehensive insights",
              },
              {
                icon: <BarChart className="h-6 w-6 text-primary" />,
                title: "Advanced Analytics",
                description:
                  "Sentiment analysis and department-wise reporting to identify areas for improvement",
              },
              {
                icon: <Video className="h-6 w-6 text-primary" />,
                title: "Video Testimonials",
                description:
                  "Capture authentic patient experiences through video feedback",
              },
              {
                icon: <Zap className="h-6 w-6 text-primary" />,
                title: "Instant Insights",
                description:
                  "Real-time analytics dashboard with sentiment and topic classification",
              },
              {
                icon: <ArrowRight className="h-6 w-6 text-primary" />,
                title: "Action Plans",
                description:
                  "Generate weekly action plans based on patient feedback for continuous improvement",
              },
              {
                icon: <MessageCircle className="h-6 w-6 text-primary" />,
                title: "Department Filtering",
                description:
                  "Filter feedback by department to identify specific areas needing attention",
              },
            ].map((feature, index) => (
              <div key={index} className="glass-card rounded-xl p-6 card-hover">
                <div className="bg-blue-50 p-3 rounded-lg inline-block mb-4">
                  {feature.icon}
                </div>
                <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
                <p className="text-gray-600">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Testimonial Section */}
      <section className="bg-gray-50 section-padding">
        <div className="container-custom">
          <div className="max-w-4xl mx-auto text-center">
            <div className="bg-white rounded-2xl shadow-sm p-8 md:p-12">
              <div className="flex justify-center mb-6">
                {[1, 2, 3, 4, 5].map((star) => (
                  <svg
                    key={star}
                    className="h-6 w-6 text-yellow-400"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                  >
                    <path d="M9.049 2.927c.3-.921 1.603-.921 1.902 0l1.07 3.292a1 1 0 00.95.69h3.462c.969 0 1.371 1.24.588 1.81l-2.8 2.034a1 1 0 00-.364 1.118l1.07 3.292c.3.921-.755 1.688-1.54 1.118l-2.8-2.034a1 1 0 00-1.175 0l-2.8 2.034c-.784.57-1.838-.197-1.539-1.118l1.07-3.292a1 1 0 00-.364-1.118L2.98 8.72c-.783-.57-.38-1.81.588-1.81h3.461a1 1 0 00.951-.69l1.07-3.292z" />
                  </svg>
                ))}
              </div>
              <blockquote className="text-xl md:text-2xl font-medium text-gray-900 mb-6">
                "CareLoop has transformed how we collect and utilize patient
                feedback. The sentiment analysis helps us quickly identify areas
                for improvement, and the weekly action plans have directly
                contributed to higher patient satisfaction scores."
              </blockquote>
              <div className="flex items-center justify-center">
                <div className="h-12 w-12 rounded-full bg-gray-200 overflow-hidden mr-4">
                  <img
                    src="https://images.unsplash.com/photo-1559839734-2b71ea197ec2?ixlib=rb-4.0.3&auto=format&fit=crop&w=100&q=80"
                    alt="Dr. Sarah Johnson"
                    className="h-full w-full object-cover"
                  />
                </div>
                <div className="text-left">
                  <p className="font-medium">Dr. Sarah Johnson</p>
                  <p className="text-gray-600 text-sm">
                    Chief Medical Officer, Northeast Hospital
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="section-padding px-36">
        <div className="container-custom">
          <div className="bg-gradient-to-r from-primary to-secondary rounded-2xl shadow-lg p-8 md:p-12 text-white text-center">
            <h2 className="heading-2 mb-4">
              Ready to improve patient satisfaction?
            </h2>
            <p className="text-lg mb-8 max-w-2xl mx-auto opacity-90">
              Join hundreds of healthcare facilities using CareLoop to
              collect and analyze patient feedback.
            </p>
            <div className="flex flex-wrap justify-center gap-4">
              <Link to="/login/admin">
                <Button
                  size="lg"
                  variant="outline"
                  className="btn-hover rounded-full px-6 bg-white border-white text-primary hover:bg-transparent"
                >
                  Get Started
                </Button>
              </Link>
              <Link to="/pricing">
                <Button
                  size="lg"
                  variant="outline"
                  className="btn-hover rounded-full px-6 border-white bg-white text-primary hover:bg-transparent"
                >
                  View Pricing
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Index;
