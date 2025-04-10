import { asyncHandler } from "../../utility/async.handler.util.js";
import { ApiErr } from "../../utility/api.error.util.js";
import { Admin } from "../../model/admin.model.js";
import { Feedback } from "../../model/feedback.model.js";

import bcryptjs from "bcryptjs";

const registration = asyncHandler(async (req, res) => {
  const { fullName, email, position, password, hospitalID } = req.body;

  const existedUser = await Admin.findOne({
    $or: [{ email }],
  });

  if (existedUser) {
    throw new ApiErr(409, "email already in use!");
  }

  const salt = await bcryptjs.genSalt(10);
  const hassedPassword = await bcryptjs.hash(password, salt);

  const newAdmin = await Admin.create({
    fullName: fullName,
    email: email,
    position: position,
    password: hassedPassword,
    hospitalID: hospitalID,
  });

  const createdNewAdmin = await Admin.findById(newAdmin._id);

  if (!createdNewAdmin) {
    throw new ApiErr(500, "err in registration!");
  }

  return res.status(201).json({
    admin: createdNewAdmin,
    msg: "successful registration!",
  });
});

const logging = asyncHandler(async (req, res) => {
  const { fullName, email, position, password, hospitalID } = req.body;

  const existedAdmin = await Admin.findOne({
    $or: [{ email }],
  });

  if (!existedAdmin) {
    throw new ApiErr(409, "no admin found!");
  }

  const isPasswordValid = await bcryptjs.compare(password, existedAdmin.password);

  if (!isPasswordValid) {
    throw new ApiErr(500, "password's wrong!");
  }

  return res.status(201).json({
    admin: existedAdmin,
    msg: "successful login!",
  });
});

const logout = asyncHandler(async (req, res) => {
  res.clearCookie("adminLoginData");

  res.status(200).json({ msg: "successful logging out!" });
});

const getAllFeedbacks = asyncHandler(async (_, res) => {
  const all = await Feedback.find();
  const positive = await Feedback.find({ sentimentIndex: 1 });
  const negative = await Feedback.find({ sentimentIndex: -1 });
  const neutral = await Feedback.find({ sentimentIndex: 0 });

  const countAll = all.length;
  const countPosetive = positive.length;
  const countNegative = negative.length;
  const countNeutral = neutral.length;

  return res.status(200).json({
    allFeedbacks: all,
    positiveFeedbacks: positive,
    negativeFeedbacks: negative,
    neutralFeedbacks: neutral,
    allCount: countAll,
    positiveCount: countPosetive,
    negativeCount: countNegative,
    neutralCount: countNeutral,
    msg: "successfully fetched feedbacks!",
  });
});

const getAllTopics = asyncHandler(async (_, res) => {
  const staff = await Feedback.find({ topic: "staff" });
  const billing = await Feedback.find({ topic: "billing" });
  const communication = await Feedback.find({ topic: "communication" });
  const waitTime = await Feedback.find({ topic: "waitTime" });
  const cleanliness = await Feedback.find({ topic: "cleanliness" });
  const foodAmenities = await Feedback.find({ topic: "foodAmenities" });
  const comfortPrivacy = await Feedback.find({ topic: "comfortPrivacy" });
  const pdc = await Feedback.find({ topic: "pdc" });
  const digitalExperience = await Feedback.find({ topic: "digitalExperience" });

  const staffCount = staff.length;
  const billingCount = billing.length;
  const communicationCount = communication.length;
  const waitTimeCount = waitTime.length;
  const cleanlinessCount = cleanliness.length;
  const foodAmenitiesCount = foodAmenities.length;
  const comfortPrivacyCount = comfortPrivacy.length;
  const pdcCount = pdc.length;
  const digitalExperienceCount = digitalExperience.length;

  return res.status(200).json({
    staffFeedbacks: staff,
    billingFeedbacks: billing,
    communicationFeedbacks: communication,
    waitTimeFeedbacks: waitTime,
    cleanlinessFeedbacks: cleanliness,
    foodAmenities: foodAmenities,
    comfortPrivacy: comfortPrivacy,
    pdc: pdc,
    digitalExperience: digitalExperience,
    staffCount: staffCount,
    billingCount: billingCount,
    communicationCount: communicationCount,
    waitTimeCount: waitTimeCount,
    cleanlinessCount: cleanlinessCount,
    foodAmenitiesCount: foodAmenitiesCount,
    comfortPrivacyCount: comfortPrivacyCount,
    pdcCount: pdcCount,
    digitalExperienceCount: digitalExperienceCount,
    msg: "successfully fetched feedbacks!",
  });
});

const getAllDepartments = asyncHandler(async (_, res) => {
  const all = await Feedback.find();
  const emergency = await Feedback.find({ departmentId: "emergency" });
  const cardiology = await Feedback.find({ departmentId: "cardiology" });
  const pediatrics = await Feedback.find({ departmentId: "pediatrics" });
  const orthopedics = await Feedback.find({ departmentId: "orthopedics" });
  const neurology = await Feedback.find({ departmentId: "neurology" });
  const oncology = await Feedback.find({ departmentId: "oncology" });
  const radiology = await Feedback.find({ departmentId: "radiology" });
  const pharmacy = await Feedback.find({ departmentId: "pharmacy" });

  const countAll = all.length;
  const countEmergency = emergency.length;
  const countCardiology = cardiology.length;
  const countPediatrics = pediatrics.length;
  const countOrthopedics = orthopedics.length;
  const countNeurology = neurology.length;
  const countOncology = oncology.length;
  const countRadiology = radiology.length;
  const countPharmacy = pharmacy.length;

  return res.status(200).json({
    allFeedbacks: countAll,
    emergency: countEmergency,
    cardiology: countCardiology,
    pediatrics: countPediatrics,
    orthopedics: countOrthopedics,
    neurology: countNeurology,
    oncology: countOncology,
    radiology: countRadiology,
    pharmacy: countPharmacy,
    msg: "successfully fetched feedbacks!",
  });
});

const GetDepartmentAnalysis = asyncHandler(async (req, res) => {
  const { deptID } = req.body;
  const feed = await Feedback.find({ departmentId: deptID });

  const positives = feed.filter((i) => i.sentimentIndex === 1);
  const negatives = feed.filter((i) => i.sentimentIndex === -1);
  const neutrals = feed.filter((i) => i.sentimentIndex === 0);

  const posetiveCount = positives.length;
  const negativeCount = negatives.length;
  const neutralCount = neutrals.length;

  return res.status(200).json({
    department: deptID,
    feed: feed,
    posetiveCount: posetiveCount,
    negativeCount: negativeCount,
    neutralCount: neutralCount,
  });
});

const respondToFeed = asyncHandler(async (req, res) => {
  const { feedbackID, response } = req.body;
  const feedback = await Feedback.findById(feedbackID);

  if (!feedback) {
    throw new ApiErr(409, "no feedback found!");
  }

  feedback.response = response;
  feedback.response_status = true;

  await feedback.save();

  return res.status(200).json({
    new: feedback,
    msg: "Feedback responded",
  });
});

const hospitalBasedFeed = asyncHandler(async (req, res) => {
  const { hospitalID } = req.body;
  const feedback = await Feedback.find({ hospitalID: hospitalID });

  return res.status(200).json({
    feed: feedback,
    msg: "Feedback recieved!",
  });
});

export {
  registration,
  logging,
  logout,
  getAllFeedbacks,
  getAllTopics,
  getAllDepartments,
  GetDepartmentAnalysis,
  respondToFeed,
  hospitalBasedFeed,
};
