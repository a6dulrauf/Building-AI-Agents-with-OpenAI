-- Helpdesk schema. Two tables, deliberately small.
--
-- CHECK constraints mirror the validation in tools.py. That duplication is
-- intentional: validation in tools.py gives the model a correctable error
-- message, while the CHECK here is the last line of defence if anything
-- ever writes to this database without going through the tools.

CREATE TABLE IF NOT EXISTS tickets (
    id             TEXT PRIMARY KEY,
    customer_email TEXT NOT NULL,
    subject        TEXT NOT NULL,
    body           TEXT NOT NULL,
    status         TEXT NOT NULL CHECK (status IN ('open', 'pending', 'resolved')),
    priority       TEXT NOT NULL CHECK (priority IN ('low', 'medium', 'high', 'urgent')),
    department     TEXT     NULL CHECK (department IS NULL OR department IN
                                        ('billing', 'technical', 'account', 'shipping')),
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS notes (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id  TEXT NOT NULL REFERENCES tickets(id),
    author     TEXT NOT NULL,
    note       TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Ticket history lookups filter by customer, so index that column.
CREATE INDEX IF NOT EXISTS idx_tickets_customer ON tickets(customer_email);
CREATE INDEX IF NOT EXISTS idx_notes_ticket ON notes(ticket_id);
