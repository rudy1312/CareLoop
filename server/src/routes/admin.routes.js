import { Router } from "express";

import {
    registration,
    logging,
    getAllFeedbacks,
    getAllTopics,
    getAllDepartments,
    GetDepartmentAnalysis
} from "../controller/admin/admin.controller.js";

const router = Router();

router.route("/reg").post(registration);
router.route("/log").post(logging);
router.route("/fetchAll").get(getAllFeedbacks);
router.route("/fetchTopics").get(getAllTopics);
router.route("/fetchDep").get(getAllDepartments);
router.route("/deptAna").post(GetDepartmentAnalysis);

export default router;