# HTTP Client in C++ for REST API Interaction

## Project Overview

This project involves building an HTTP client in C++ that interacts with a REST API exposed by a server. The client communicates with the server over HTTP to perform various operations, simulating an online library. The project helps reinforce fundamental concepts in web architecture, specifically around the HTTP protocol and RESTful services.

## Objectives

- Understand the mechanisms of the HTTP protocol.
- Interact with a REST API and work with HTTP methods.
- Familiarize with web concepts like JSON, session management, and JWT (JSON Web Token).
- Utilize external libraries for JSON manipulation and REST API interactions.

## Summary

The goal is to develop an HTTP client in C++ that sends requests to a REST API representing an online library. The server backend is already implemented and simulates a library system, responding to HTTP requests made by the client.

The client will function as a command-line interface (CLI) where users can interact with the library by executing commands.

## Server Details

### Connection Information

- **HOST**: 34.246.184.49
- **PORT**: 8080

### Server API Endpoints

The server allows the following actions:

1. **User Registration**

   - **Endpoint**: `POST /api/v1/tema/auth/register`
   - **Payload Type**: `application/json`
   - **Payload Structure**:
     ```json
     {
       "username": "String",
       "password": "String"
     }
     ```
   - **Error Handling**:
     - Returns an error if the username is already taken.

2. **User Authentication**

   - **Endpoint**: `POST /api/v1/tema/auth/login`
   - **Payload Type**: `application/json`
   - **Payload Structure**:
     ```json
     {
       "username": "String",
       "password": "String"
     }
     ```
   - **Response**:
     - Returns a session cookie on successful login.
   - **Error Handling**:
     - Returns an error if credentials are incorrect.

3. **Library Access Request**

   - **Endpoint**: `GET /api/v1/tema/library/access`
   - **Additional Information**:
     - Requires authentication (session cookie).
   - **Response**:
     - Returns a session cookie on successful access request.
   - **Error Handling**:
     - Returns an error if authentication fails.

4. **View Summary of All Books**

   - **Endpoint**: `GET /api/v1/tema/library/books`
   - **Additional Information**:
     - Requires library access (session cookie).
   - **Response**:
     - Returns a summary of all available books in the library.

## Client Implementation

The client is a command-line application that sends requests to the server based on user input. Depending on the command entered, it constructs and sends the appropriate HTTP request and processes the server’s response.

### Commands and Actions

- **register**: Registers a new account by sending a `POST` request with `username` and `password`.
- **login**: Authenticates a user by sending a `POST` request with `username` and `password`.
- **access**: Requests access to the library by sending a `GET` request.
- **view_books**: Retrieves and displays a summary of all books by sending a `GET` request.

### Error Handling

The client handles various error scenarios based on the server’s responses, including:
- Duplicate usernames during registration.
- Incorrect credentials during login.
- Access denial when attempting library actions without proper authentication.

### External Libraries

The client leverages external libraries for:
- **JSON Parsing**: To handle JSON data in payloads and server responses.
- **HTTP Requests**: To manage communication over HTTP with the server.

## Project Setup and Execution

1. **Compile the Client**:
   - Ensure necessary libraries for JSON and HTTP are installed.
   - Compile the client using a C++ compiler.

2. **Run the Client**:
   - Execute the client program and follow the command prompts to interact with the server API.

## Notes

- **Authentication**: Ensure to manage session cookies properly for authenticated requests.
- **Payload Formatting**: Follow the JSON structure exactly for successful API interaction.
- **Error Handling**: Be vigilant about handling errors and unexpected responses from the server.

This project enhances understanding of client-server interactions over HTTP and offers hands-on experience with RESTful APIs and JSON data handling.
