import express from "express";
import cors from "cors";
import serverless from "serverless-http";

const app = express();

app.use(cors());

app.use(express.json());

import adminRoutes from "./routes/admin.routes.js";
import patientRoutes from "./routes/patient.routes.js";

app.get("/", (req, res) => {
  res.send("API is running...");
});

app.use("/bloom/v1/api/admin", adminRoutes);
app.use("/bloom/v1/api/patient", patientRoutes);

export {app};
