import { Router } from "express";

import {
    registration,
    logging,
    logout,
    getAllFeedbacks,
    getAllTopics,
    getAllDepartments,
    GetDepartmentAnalysis,
    respondToFeed,
    hospitalBasedFeed
} from "../controller/admin/admin.controller.js";

const router = Router();

router.route("/reg").post(registration);
router.route("/log").post(logging);
router.route("/out").post(logout);
router.route("/fetchAll").get(getAllFeedbacks);
router.route("/fetchTopics").get(getAllTopics);
router.route("/fetchDep").get(getAllDepartments);
router.route("/deptAna").post(GetDepartmentAnalysis);
router.route("/hospital-feed").post(hospitalBasedFeed);
router.route("/respond").post(respondToFeed);

export default router;