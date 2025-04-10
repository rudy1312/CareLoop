import express from "express";
import cors from "cors";
import serverless from "serverless-http";

const app = express();

// app.use(
//   cors({
//     origin: "http://localhost:8080",
//     credentials: true,
//   })
// );

const allowedOrigins = [
  "http://localhost:8080",  // for development
  "https://care-loop.vercel.app", // for production
];

app.use(cors({
  origin: function (origin, callback) {
    if (!origin || allowedOrigins.includes(origin)) {
      callback(null, true);
    } else {
      callback(new Error("Not allowed by CORS"));
    }
  },
  credentials: true,
}));


app.use(express.json());

import adminRoutes from "./routes/admin.routes.js";
import patientRoutes from "./routes/patient.routes.js";

app.get("/", (req, res) => {
  res.send("API is running...");
});

app.use("/bloom/v1/api/admin", adminRoutes);
app.use("/bloom/v1/api/patient", patientRoutes);

export {app};
