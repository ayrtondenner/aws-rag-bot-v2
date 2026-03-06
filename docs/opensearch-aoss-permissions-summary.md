# OpenSearch Serverless (AOSS) permissions — step-by-step setup (AWS Bot summary)

Below is the “order of operations” we converged on for **OpenSearch Serverless (AOSS)**, based on our earlier troubleshooting (especially the *“why can’t I create an index / ML connector?”* rabbit hole) + what’s echoed in your audio notes (user vs role vs collection policy vs root confusion).

---

## Mental model (why it felt so painful)

In AOSS, **IAM permissions are not enough**. You usually need *all three* aligned:

1) **IAM identity policy** (attached to user/role): lets the principal call AOSS APIs (control-plane + some data-plane actions).  
2) **AOSS Data Access Policy** (on the collection): *actually grants access* to the collection/index/model resources for specific principals.  
3) **Network/Security policies** (encryption + network): decides if the endpoint is reachable.

And a big gotcha we hit: **the principal you’re logged in with (Console/CloudShell) is often NOT the same principal you granted in the collection policy**, so index creation fails until you add the *real* ARN (sometimes even `...:root`) into the Data Access Policy principals.

---

## Step-by-step setup (do this BEFORE “configuring OpenSearch”)

### Step 0) Decide the principals you will use (this prevents the “root vs user” trap)

Make a short list of **every identity that will touch the collection**, for example:

- **Console admin principal** (SSO role / admin role / whatever you use in the AWS console)
- **API principal** (your IAM user like `BedrockAPIKey-wzyn`)
- **Connector execution role** (role that OpenSearch/ML assumes to call Bedrock)

Then, whenever you test anything from CloudShell, run:

- `aws sts get-caller-identity`

Whatever ARN appears there **must be included** in the collection’s Data Access Policy principals if you expect it to create indexes, connectors, etc. This was the key “why is it denying me?” moment.

---

### Step 1) Create / confirm the API IAM User (your “BedrockAPIKey-wzyn”)

**Purpose:** programmatic access (Python ingestion, queries).

Attach (minimum):

- Permissions to call AOSS (serverless) APIs you need
- Permissions to call Bedrock (invoke embedding / rerank / LLM if your app does that)
- **Later:** permission to `iam:PassRole` (only if you’ll create ML connectors that require a role)

> In our notes: create user → attach policy → but it wasn’t enough until roles + collection policy principals were aligned.

---

### Step 2) Create the Connector Role (the role OpenSearch ML will assume)

**Purpose:** OpenSearch uses this role to call **Bedrock** (or other model endpoints) when creating/using an ML connector.

You need:

1) **Permissions policy on the role** (example: allow `bedrock:InvokeModel` for the models you’ll use)  
2) **Trust policy on the role** so OpenSearch ML can assume it (we used the ML service principal in the trust)  
3) Your human/API principal must be able to **pass** this role:
   - `iam:PassRole` on that connector role ARN

This matches the “role has permissions policy + trust policy” realization.

---

### Step 3) Create / confirm the AOSS collection “infrastructure” policies

These are prerequisites around the collection:

- **Encryption policy** (KMS key or AWS-owned key)
- **Network policy** (public + allowlist, or VPC endpoints)

If the network policy blocks you, Dashboards and API calls will look like “random failures”.

---

### Step 4) Create/Update the Data Access Policy on the collection (this is the big one)

This policy must include **ALL principals** from Step 0, and cover the **resource types** you’ll touch.

#### 4.1 Collection permissions (minimum to manage collection items)

You already had something like:

- `aoss:CreateCollectionItems`, `aoss:DeleteCollectionItems`, `aoss:UpdateCollectionItems`, `aoss:DescribeCollectionItems`

#### 4.2 Index permissions (required to create the vector index + read/write documents)

Add an **index rule** (commonly needed):

- create/describe/update/delete index
- read/write documents

(Exact permission names vary by the AOSS console’s options, but the point is: don’t stop at *collection* permissions if you’re creating indexes.)

#### 4.3 ML permissions (fixes “Access denied to create ML connector”)

When you create a vector index in the console (especially with “semantic/ML” features), AOSS may try to create/use an **ML connector**. If your policy only covered `ResourceType: collection`, connector creation can fail.

Add a rule like:

```json
{
  "ResourceType": "model",
  "Resource": ["model/ragbot-v2-collection/*"],
  "Permission": [
    "aoss:DescribeMLResource",
    "aoss:CreateMLResource",
    "aoss:UpdateMLResource",
    "aoss:DeleteMLResource",
    "aoss:ExecuteMLResource"
  ]
}
```

#### 4.4 Principals: include user + role + the actual console identity

This was the painful bit: **it wasn’t enough to list only the user or only the role**; you typically need both, and also the identity you were *actually logged in with* (sometimes `root` shows up in CloudShell).

---

### Step 5) Only now: create the vector index / connector / dashboards setup

Once steps 0–4 are done:

1) Open the collection → create the **vector index**
2) If using “OpenSearch calls the model”:
   - create the **ML connector** (or let the console do it)
   - ensure:
     - the Data Access Policy has **model** permissions
     - the creating principal has **iam:PassRole**
     - the connector role trust policy is correct
3) Open **Dashboards**
   - if it denies access, it’s usually either:
     - the principal ARN missing from Data Access Policy, or
     - missing Dashboards permission in IAM, or
     - network policy blocking

---

## Quick verification checklist (fastest way to avoid looping)

- In the same environment you’re using (Console/CloudShell/your machine):
  - `aws sts get-caller-identity` → copy the ARN
- In the AOSS **Data Access Policy principals**, confirm it includes:
  - that ARN (or its role)
  - your API user ARN
  - your connector role ARN (if applicable)
- If creating ML connector:
  - Data Access Policy includes **ResourceType: model**
  - Your creator principal has **iam:PassRole** to the connector role

---

## What the audio notes were capturing (in plain terms)

- “I created a user + policy… didn’t work… had to create a role… role has trust policy… then collection has another policy… had to list both user and role… and also root because that’s who I was logged in as.”
- “It’s confusing because the admin who creates things isn’t automatically granted data access unless you explicitly add that admin principal to the collection policy.”
