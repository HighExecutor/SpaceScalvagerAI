using System.Collections;
using System.Collections.Generic;
using System.Numerics;
using UnityEngine;
using UnityEngine.InputSystem;
using Quaternion = UnityEngine.Quaternion;
using Vector2 = UnityEngine.Vector2;
using Vector3 = UnityEngine.Vector3;

public class SpaceShipController : MonoBehaviour
{
    public float velocity;
    public float angularVelocity;
    private Vector3 movementInput;
    private Vector2 rotateInput;
    private Rigidbody rb;
    private ParticleSystem parts;
    private Transform body;
    public GameObject aimPosition;
    public GameObject[] lasers;
    public TrailRenderer railgunTrail;
    public float shootCooldown;
    private float curShootCooldown;
    public float shootRange;
    private bool canShoot;
    public ParticleSystem shootEffect;
    public CargoUIScript cargoUI;

    private float curMinerals;
    public float maxMinerals;


    // Start is called before the first frame update
    void Start()
    {
        rb = GetComponent<Rigidbody>();
        parts = GetComponentInChildren<ParticleSystem>();
        body = transform.GetChild(0);
        curShootCooldown = 0.0f;
        canShoot = true;
        curMinerals = 0.0f;
        cargoUI.SetMaxCargo((int)maxMinerals);
        cargoUI.SetText(0.0f);
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        Move();
        UpdateCooldown();
    }

    private void UpdateCooldown()
    {
        if (curShootCooldown > 0)
        {
            curShootCooldown -= Time.fixedDeltaTime;
            if (curShootCooldown <= 0)
            {
                canShoot = true;
                curShootCooldown = 0.0f;
            }
        }
    }

    private void Move()
    {
        // move
        if (movementInput != Vector3.zero)
        {
            Vector3 forward = transform.forward * movementInput.z;
            Vector3 left = transform.right * movementInput.x;
            Vector3 up = transform.up * movementInput.y;
            Vector3 moveVector = (forward + left + up).normalized;
            rb.AddForce(moveVector * velocity * Time.fixedDeltaTime);
            
        }
        float bodyAngle = body.rotation.eulerAngles.z;
        if (bodyAngle > 180)
        {
            bodyAngle -= 360;
        }
        if (bodyAngle < -180)
        {
            bodyAngle += 360;
        }
        if (movementInput.x != 0)
        {
            bodyAngle -= Mathf.Sign(movementInput.x) * 35 * Time.fixedDeltaTime;
            bodyAngle = Mathf.Clamp(bodyAngle, -15, 15);
            body.rotation = Quaternion.Euler(body.rotation.eulerAngles.x, body.rotation.eulerAngles.y, bodyAngle);
        } else if (body.rotation.eulerAngles.z != 0)
        {
            bodyAngle -= Mathf.Sign(bodyAngle) * 35 * Time.fixedDeltaTime;
            bodyAngle = Mathf.Clamp(bodyAngle, -15, 15);
            if (Mathf.Abs(bodyAngle) < 2)
            {
                bodyAngle = 0;
            }
            body.rotation = Quaternion.Euler(body.rotation.eulerAngles.x, body.rotation.eulerAngles.y, bodyAngle);
        }
        if (movementInput.z > 0)
        {
            parts.Play();
        }
        else if (parts.isPlaying)
        {
            parts.Pause();
            parts.Clear();
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
        if (rotateInput.x != 0 || rotateInput.y != 0 || curRotation.z != 0)
        {
            float horAngle = curRotation.y + rotateInput.x * angularVelocity * Time.fixedDeltaTime;
            float vertAngle = curRotation.x + rotateInput.y * angularVelocity * Time.fixedDeltaTime;
            vertAngle = Mathf.Clamp(vertAngle, -45.0f, 45.0f);
            float diagAngle = curRotation.z;
            if (diagAngle != 0)
            {
                if (diagAngle > 180)
                {
                    diagAngle -= 360;
                }

                if (diagAngle < -180)
                {
                    diagAngle += 360;
                }

                diagAngle -= Mathf.Sign(diagAngle) * angularVelocity * Time.fixedDeltaTime;
                if (Mathf.Abs(diagAngle) < 2)
                {
                    diagAngle = 0;
                }
            }
            transform.rotation = Quaternion.Euler(vertAngle, horAngle, diagAngle);
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

    void OnShoot()
    {
        if (canShoot)
        {
            Vector3 direction = transform.forward.normalized;
            float range = shootRange;
            LayerMask mask = LayerMask.GetMask("Objects");
            if (Physics.Raycast(aimPosition.transform.position, direction, out RaycastHit hit, shootRange, mask))
            {
                Debug.Log("Hit distance" + hit.distance);
                range = Vector3.Distance(aimPosition.transform.position, hit.point);
                ParticleSystem hitEffect = Instantiate(shootEffect, hit.point, Quaternion.identity);
                Destroy(hitEffect.gameObject, 0.5f);
                if (hit.collider.CompareTag("Meteor"))
                {
                    MeteorScript meteor = hit.collider.GetComponent<MeteorScript>();
                    meteor.TakeHit();
                }
            }
            for (int l = 0; l < lasers.Length; l++)
            {
                TrailRenderer trail = Instantiate(railgunTrail, lasers[l].transform);
                StartCoroutine(SpawnTrail(trail, range, shootRange));

            }
            canShoot = false;
            curShootCooldown = shootCooldown;
        }
    }

    IEnumerator SpawnTrail(TrailRenderer trail, float range, float maxRange)
    {
        float time = 0f;
        Vector3 startPosition = trail.transform.position;
        Vector3 endPoint = startPosition + trail.transform.up.normalized * range;
        float rangeRate = range / maxRange;
        while (time < rangeRate)
        {
            trail.transform.position = Vector3.Lerp(startPosition, endPoint, time / rangeRate);
            time += Time.fixedDeltaTime / trail.time;
            yield return null;
        }
        trail.transform.position = endPoint;
        Destroy(trail.gameObject, trail.time);

    }

    public void TakeMineral(float amount)
    {
        curMinerals = Mathf.Min(maxMinerals, curMinerals + amount);
        cargoUI.SetCargo(curMinerals);
        cargoUI.SetText(curMinerals);
        
    }
    
}