using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.InputSystem;

public class SpaceShipController : MonoBehaviour
{
    public float velocity;
    public float angularVelocity;
    private Vector3 movementInput;
    private Vector2 rotateInput;
    private Rigidbody rb;


    // Start is called before the first frame update
    void Start()
    {
        rb = GetComponent<Rigidbody>();
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        // move
        Debug.DrawLine(transform.position, transform.position+transform.forward.normalized*10, Color.green);
        Debug.DrawLine(transform.position, transform.position+transform.right.normalized*10, Color.blue);
        Debug.DrawLine(transform.position, transform.position+transform.up.normalized*10, Color.yellow);
        if (movementInput != Vector3.zero)
        {
            Vector3 forward = transform.forward * movementInput.z;
            Vector3 left = transform.right * movementInput.x;
            Vector3 up = transform.up * movementInput.y;
            Vector3 moveVector = (forward + left + up).normalized;
            rb.AddForce(moveVector * velocity * Time.fixedDeltaTime);
        }

        Vector3 curRotation = transform.rotation.eulerAngles;
        if (curRotation.x > 180)
        {
            curRotation.x -= 360;
        }
        if (curRotation.x < -180)
        {
            curRotation.x += 360;
        } 
        if (rotateInput.x != 0 || rotateInput.y != 0)
        {
            float horAngle = curRotation.y + rotateInput.x * angularVelocity * Time.fixedDeltaTime;
            float vertAngle = curRotation.x + rotateInput.y * angularVelocity * Time.fixedDeltaTime;
            vertAngle = Mathf.Clamp(vertAngle, -45.0f, 45.0f);
            transform.rotation = Quaternion.Euler(vertAngle, horAngle, curRotation.z);
        }
    }

    void OnMove(InputValue inputValue)
    {
        movementInput = inputValue.Get<Vector3>();
    }

    void OnRotate(InputValue inputValue)
    {
        rotateInput = inputValue.Get<Vector2>();
    }
}