import { Router } from "express";

import {
    registration,
    logging,
    feeding
} from "../controller/patient/patient.controller.js";

const router = Router();

router.route("/reg").post(registration);
router.route("/log").post(logging);
router.route("/feed").post(feeding);

export default router;