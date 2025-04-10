import { asyncHandler } from "../../utility/async.handler.util.js";
import { ApiErr } from "../../utility/api.error.util.js";
import { Patient } from "../../model/patient.model.js";
import { Feedback } from "../../model/feedback.model.js";

const registration = asyncHandler(async (req, res) => {
  const { fullName, patientID, mobile, hospitalID } = req.body;

  const existedUser = await Patient.findOne({
    $or: [{ mobile }],
  });

  if (existedUser) {
    throw new ApiErr(409, "mobile number already in use!");
  }

  const newPatient = await Patient.create({
    fullName: fullName,
    mobile: mobile,
    patientID: patientID,
    hospitalID: hospitalID,
  });

  const createdNewPatient = await Patient.findById(newPatient._id);

  if (!createdNewPatient) {
    throw new ApiErr(500, "err in registration!");
  }

  return res.status(201).json({
    patient: createdNewPatient,
    msg: "successful registration!",
  });
});

const logging = asyncHandler(async (req, res) => {
  const { fullName, patientID, mobile, hospitalID } = req.body;

  const existedPatient = await Patient.findOne({
    $or: [{ patientID }],
  });

  if (!existedPatient) {
    throw new ApiErr(409, "no user found!");
  }

  if (!fullName === existedPatient.fullName) { 
    throw new ApiErr(500, "name's wrong!");
  }

  if (!mobile === existedPatient.mobile) {
    throw new ApiErr(500, "mobile's wrong!");
  }

  if (!hospitalID === existedPatient.hospitalID) {
    throw new ApiErr(500, "hospital's wrong!");
  }

  return res.status(201).json({
    patient: existedPatient,
    msg: "successful login!",
  });
});

const logout = asyncHandler(async (req, res) => {
  res.clearCookie("patientData");

  return res.status(201).json({
    msg: "successful logging out!",
  });
});

const feeding = asyncHandler(async (req, res) => {
  const {
    patientID,
    hospitalID,
    departmentId,
    sentimentIndex,
    topic,
    contentTypeIndex,
    textContent,
    mediaContent,
  } = req.body;

  const newFeedback = await Feedback.create({
    patientID,
    hospitalID,
    departmentId,
    sentimentIndex,
    topic,
    contentTypeIndex,
    textContent,
    mediaContent,
  });

  const newCreatedFeedback = await Feedback.findById(newFeedback._id);

  if (!newCreatedFeedback) {
    throw new ApiErr(500, "err in submission!");
  }

  return res.status(201).json({
    feedback: newCreatedFeedback,
    msg: "successful submission!",
  });
});

const getMyFeed = asyncHandler(async (req, res) => {
  const { patientID } = req.body;
  const myFeed = await Feedback.find({ patientID: patientID });
  const count = myFeed.length;

  return res.status(200).json({
    count: count,
    my_feedbacks: myFeed,
    msg: "successfully fetched feedbacks!",
  });
});

export { registration, logging, logout, feeding, getMyFeed };
