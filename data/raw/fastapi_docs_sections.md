# FastAPI Documentation — Source Sections

Raw sections used as context for the QA dataset.
Source: https://fastapi.tiangolo.com/tutorial/

---

## Query Parameters

When you declare other function parameters that are not part of the path
parameters, they are automatically interpreted as query parameters.
The query is the set of key-value pairs that go after the ? in a URL,
separated by & characters. For example, in the URL:
http://127.0.0.1:8000/items/?skip=0&limit=10
the query parameters are: skip with a value of 0, and limit with a value of 10.
As they are part of the URL, they are naturally strings. But when you declare
them with Python types (in the example above, as int), they are converted to
that type and validated against it. As query parameters are not a fixed part
of a path, they can be optional and can have default values.
All the same process that applied for path parameters also applies for query
parameters: Editor support, Data parsing, Data validation, Automatic documentation.
But when you want to make a query parameter required, you can just not declare
any default value.

---

## Path Parameters

You can declare the type of a path parameter in the function, using standard
Python type annotations. In this case, item_id is declared to be an int.
This will give you editor support inside of your function, with error checks,
completion, etc. Notice that the value your function received (and returned)
is 3, as a Python int, not a string '3'.

You can declare multiple path parameters and query parameters at the same time,
FastAPI knows which is which. And you don't have to declare them in any specific
order. FastAPI will detect the parameters by their names, types and default
declarations (Query, Path, etc), it doesn't care about the order.

You can declare a path parameter containing a path using a URL like:
/files/{file_path:path}. In this case, the name of the parameter is file_path,
and the last part, :path, tells it that the parameter should match any path.
You might need the parameter to contain /home/johndoe/myfile.txt, with a leading
slash (/). In that case, the URL would be: /files//home/johndoe/myfile.txt,
with a double slash (//) between files and home.

---

## Query Validation

You can pass more parameters to Query. In this case, the max_length parameter
that applies to strings: q: str | None = Query(default=None, max_length=50).
This will validate the data, show a clear error when the data is not valid,
and document the parameter in the OpenAPI schema path operation.

---

## OpenAPI Schema

FastAPI generates a schema with all your API using the OpenAPI standard for
defining APIs. A schema is a definition or description of something. Not the
code that implements it, but just an abstract description. In this case,
OpenAPI is a specification that dictates how to define a schema of your API.
This schema definition includes your API paths, the possible parameters they
take, etc.

---

## HTTP Methods

In the HTTP protocol, you can communicate to each path using one or more of
these methods. When building APIs, you normally use these specific HTTP methods
to perform a specific action: POST to create data, GET to read data, PUT to
update data, DELETE to delete data. So, in OpenAPI, each of the HTTP methods
is called an operation.

---

## Body Parameters

Let's say you only have a single item body parameter from a Pydantic model Item.
By default, FastAPI will then expect its body directly. But if you want it to
expect a JSON with a key item and inside of it the model contents, as it does
when you declare extra body parameters, you can use the special Body parameter embed.
