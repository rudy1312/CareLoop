import { Router } from "express";

import {
    registration,
    logging,
    logout,
    feeding,
    getMyFeed
} from "../controller/patient/patient.controller.js";

const router = Router();

router.route("/reg").post(registration);
router.route("/log").post(logging);
router.route("/out").post(logout);
router.route("/feed").post(feeding);
router.route("/fetchMy").post(getMyFeed);

export default router;