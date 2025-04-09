import mongoose, { Schema } from "mongoose";

const adminSchema = new Schema({
    fullName: {
        type: String
    },
    email: {
        type: String
    },
    position: {
        type: String
    },
    password: {
        type: String
    },
    hospitalID: {
        type: String
    }
}, { timestamps: true });

export const Admin = mongoose.model("Admin", adminSchema);