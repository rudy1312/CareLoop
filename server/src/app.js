import express from "express";
const app = express();

app.use(express.json());

import adminRoutes from "./routes/admin.routes.js"
import patientRoutes from "./routes/patient.routes.js"

app.use("/bloom/v1/api/admin", adminRoutes);
app.use("/bloom/v1/api/patient", patientRoutes);

export { app };