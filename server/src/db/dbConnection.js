import mongoose from "mongoose";

const connectDb = async () => {
    try {
        const dbURL = process.env.DB_URL;
        if (!dbURL) {
            throw new Error("Database URL is not defined in environment variables");
        }
        const connect = await mongoose.connect(dbURL);
        console.log(`Connection Established -> ${connect.connection.host}`);        
    }catch(err) {
        console.log("Error in connection", err);
        process.exit(1);        
    }
};

export default connectDb;