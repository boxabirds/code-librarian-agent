## Codebase Knowledge Checkpoint

### Overview

The codebase appears to be a full-stack web application, possibly a CMS or a tool for managing landscapes and websites. It utilizes React for the frontend, likely with TypeScript, and seems to have a backend built with Express.js (inferred from the presence of server/routes.ts and server/index.ts). It also uses Drizzle ORM for database interactions.

### Key Components

*   **`client/`**: Contains the frontend code.
    *   **`client/src/App.tsx`**: The main application component.
    *   **`client/src/components/`**: Reusable UI components.
        *   `client/src/components/landscape-form.tsx`: Component for landscape-related forms.
        *   `client/src/components/website-form.tsx`: Component for website-related forms.
        *   `client/src/components/ui/`: UI primitives built with Radix UI and Tailwind CSS.
    *   **`client/src/hooks/`**: Custom React hooks.
        *   `client/src/hooks/use-websites.ts`: Hook for managing website data.
        *   `client/src/hooks/useEnrichment.ts`: Hook related to data enrichment.
    *   **`client/src/lib/`**: Utility functions.
        *   `client/src/lib/csv-utils.ts`: Utilities for CSV file handling.
        *   `client/src/lib/url-utils.ts`: Utilities for URL manipulation.
    *   **`client/src/pages/`**: React Router pages.
        *   `client/src/pages/landscapes.tsx`: Page for managing landscapes.
        *   `client/src/pages/websites.tsx`: Page for managing websites.
    *   **`client/src/workers/`**: Web workers for background tasks.
        *   `client/src/workers/enrichment.worker.ts`: Worker for data enrichment tasks.
*   **`db/`**: Contains database schema and related files.
    *   `db/schema.ts`: Defines the database schema using Drizzle ORM.
    *   `db/types.ts`: Defines TypeScript types related to the database schema.
*   **`server/`**: Contains the backend code.
    *   `server/handlers/`: Request handlers for different routes.
        *   `server/handlers/websites.ts`: Handlers for website-related routes.
        *   `server/handlers/landscapes.ts`: Handlers for landscape-related routes.
        *   `server/handlers/enrich.ts`: Handlers for enrichment-related routes.
    *   `server/routes.ts`: Defines the API routes.
    *   `server/index.ts`: Entry point for the backend server.
*   **`public/uploads/`**: Directory for storing uploaded files, likely website icons.
*   **`migrations/`**: Database migration files.

### Technologies Used

*   **Frontend**: React, TypeScript, Radix UI, Tailwind CSS, potentially Vite.
*   **Backend**: Express.js (inferred), TypeScript, Drizzle ORM.
*   **Database**: (Not explicitly specified, but Drizzle ORM supports PostgreSQL, MySQL, SQLite, etc.)

### Key Functionality

*   **Landscape Management**: Allows users to manage landscape data.
*   **Website Management**: Allows users to manage website data.
*   **Data Enrichment**: Includes functionality for enriching data, potentially using web workers for background processing.
*   **CSV Upload**: Supports uploading data from CSV files.

### Potential Areas of Interest

*   The interaction between the frontend and backend for data fetching and manipulation.
*   The implementation of the data enrichment process using web workers.
*   The database schema and how it relates to the application's data model.
*   The UI components and how they are used to build the user interface.
*   The routing logic in both the frontend and backend.

### Notes

*   The presence of `theme.json` suggests the use of a themeing library or system.
*   The `install-dev-mac.sh` script indicates a development setup for macOS.
*   The large number of icon files in `public/uploads/` suggests that the application allows users to upload website icons.