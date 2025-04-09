import { Clock, Heart, Shield, Users } from "lucide-react";
import { Link } from "react-router-dom";
import { Button } from "@/components/ui/button";

const AboutPage = () => {
  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="bg-gradient-to-br from-blue-50 to-cyan-50 section-padding">
        <div className="container-custom">
          <div className="max-w-3xl mx-auto text-center space-y-6">
            <h1 className="heading-1 text-gray-900">
              Our mission is to improve healthcare through feedback
            </h1>
            <p className="text-lg text-gray-600">
              CareLoop was founded with a simple but powerful idea: give
              patients a voice and help healthcare providers listen and improve.
            </p>
          </div>
        </div>
      </section>

      {/* Our Story Section */}
      <section className="section-padding">
        <div className="container-custom">
          <div className="grid md:grid-cols-2 gap-12 items-center px-20">
            <div>
              <img
                src="https://images.unsplash.com/photo-1576091160550-2173dba999ef?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80"
                alt="Team meeting"
                className="rounded-2xl shadow-lg"
              />
            </div>
            <div className="space-y-6">
              <h2 className="heading-2 text-gray-900">Our Story</h2>
              <p className="text-gray-600">
                Founded in 2020 by a team of healthcare professionals and
                technology experts, CareLoop was born from firsthand
                experience with the challenges of collecting and analyzing
                patient feedback.
              </p>
              <p className="text-gray-600">
                We noticed that traditional surveys were limiting and often
                failed to capture the nuanced experiences of patients. By
                enabling feedback in multiple formats - text, voice, and video -
                we've created a platform that lets patients express themselves
                naturally.
              </p>
              <p className="text-gray-600">
                Today, CareLoop is used by hundreds of healthcare facilities
                worldwide, helping them improve patient care through actionable
                insights.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Values Section */}
      <section className="bg-gray-50 section-padding">
        <div className="container-custom">
          <div className="text-center max-w-3xl mx-auto mb-16">
            <h2 className="heading-2 text-gray-900 mb-4">Our Values</h2>
            <p className="text-lg text-gray-600">
              These core principles guide everything we do at CareLoop
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 px-20">
            {[
              {
                icon: <Users className="h-10 w-10 text-primary" />,
                title: "Patient-Centered",
                description:
                  "We put patients at the center of everything we build, ensuring their voices are heard",
              },
              {
                icon: <Shield className="h-10 w-10 text-primary" />,
                title: "Data Privacy",
                description:
                  "We maintain the highest standards of security and privacy for sensitive healthcare data",
              },
              {
                icon: <Heart className="h-10 w-10 text-primary" />,
                title: "Empathy",
                description:
                  "We design with empathy for both patients and healthcare providers",
              },
              {
                icon: <Clock className="h-10 w-10 text-primary" />,
                title: "Continuous Improvement",
                description:
                  "We constantly iterate and improve our platform based on user feedback",
              },
            ].map((value, index) => (
              <div
                key={index}
                className="bg-white rounded-xl p-8 shadow-sm text-center"
              >
                <div className="bg-blue-50 p-4 rounded-full inline-block mb-6">
                  {value.icon}
                </div>
                <h3 className="text-xl font-semibold mb-3">{value.title}</h3>
                <p className="text-gray-600">{value.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Team Section */}
      <section className="section-padding">
        <div className="container-custom">
          <div className="text-center max-w-3xl mx-auto mb-16">
            <h2 className="heading-2 text-gray-900 mb-4">
              Our Leadership Team
            </h2>
            <p className="text-lg text-gray-600">
              Meet the people driving our mission forward
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 px-20">
            {[
              {
                name: "Dr. Emily Chen",
                role: "CEO & Co-Founder",
                bio: "Former hospital administrator with 15+ years in healthcare management",
                image:
                  "https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?ixlib=rb-4.0.3&auto=format&fit=crop&w=300&q=80",
              },
              {
                name: "Michael Rodriguez",
                role: "CTO & Co-Founder",
                bio: "Software engineer with expertise in healthcare technology and data security",
                image:
                  "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?ixlib=rb-4.0.3&auto=format&fit=crop&w=300&q=80",
              },
              {
                name: "Dr. Sarah Johnson",
                role: "Chief Medical Officer",
                bio: "Board-certified physician focused on improving patient care through technology",
                image:
                  "https://images.unsplash.com/photo-1559839734-2b71ea197ec2?ixlib=rb-4.0.3&auto=format&fit=crop&w=300&q=80",
              },
              {
                name: "David Park",
                role: "VP of Product",
                bio: "Product leader with experience building healthcare applications",
                image:
                  "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-4.0.3&auto=format&fit=crop&w=300&q=80",
              },
              {
                name: "Lisa Thompson",
                role: "VP of Customer Success",
                bio: "Healthcare consultant specializing in patient experience and satisfaction",
                image:
                  "https://images.unsplash.com/photo-1580489944761-15a19d654956?ixlib=rb-4.0.3&auto=format&fit=crop&w=300&q=80",
              },
              {
                name: "James Wilson",
                role: "VP of Sales",
                bio: "Experienced sales leader in healthcare SaaS with a background in nursing",
                image:
                  "https://images.unsplash.com/photo-1519085360753-af0119f7cbe7?ixlib=rb-4.0.3&auto=format&fit=crop&w=300&q=80",
              },
            ].map((person, index) => (
              <div
                key={index}
                className="glass-card rounded-xl overflow-hidden card-hover"
              >
                <div className="h-48 overflow-hidden">
                  <img
                    src={person.image}
                    alt={person.name}
                    className="w-full h-full object-cover"
                  />
                </div>
                <div className="p-6">
                  <h3 className="text-xl font-semibold">{person.name}</h3>
                  <p className="text-primary font-medium text-sm mb-2">
                    {person.role}
                  </p>
                  <p className="text-gray-600">{person.bio}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="bg-gradient-to-br from-blue-50 to-cyan-50 section-padding">
        <div className="container-custom text-center">
          <div className="max-w-3xl mx-auto space-y-6">
            <h2 className="heading-2 text-gray-900">
              Join us in transforming patient feedback
            </h2>
            <p className="text-lg text-gray-600">
              Experience how CareLoop can help your healthcare facility
              improve patient satisfaction and outcomes.
            </p>
            <div className="pt-4 flex justify-center gap-4">
              <Link to="/login/admin">
                <Button
                  variant="outline"
                  size="lg"
                  className="btn-hover rounded-full px-6 hover:bg-cyan-500"
                >
                  Get Started
                </Button>
              </Link>
              <Link to="/contact">
                <Button
                  variant="outline"
                  size="lg"
                  className="btn-hover rounded-full px-6 hover:bg-cyan-500"
                >
                  Contact Us
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default AboutPage;
