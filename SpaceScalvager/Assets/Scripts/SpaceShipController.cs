using System;
using System.Collections;
using System.Collections.Generic;
using System.Numerics;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.InputSystem;
using Quaternion = UnityEngine.Quaternion;
using Vector2 = UnityEngine.Vector2;
using Vector3 = UnityEngine.Vector3;

public class SpaceShipController : Agent
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
    private SpaceManager spaceManager;
    private float curMinerals;
    public float maxMinerals;

    private Vector3 startPosition;
    private Quaternion startRotate;


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
        cargoUI.SetCargo(0.0f);
        startPosition = transform.position;
        startRotate = transform.rotation;
        spaceManager = GetComponentInParent<SpaceManager>();
    }

    void CustomFixedUpdate()
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

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        float x = actionBuffers.ContinuousActions[0];
        float y = actionBuffers.ContinuousActions[1];
        float z = actionBuffers.ContinuousActions[2];
        float qx = actionBuffers.ContinuousActions[3];
        float qy = actionBuffers.ContinuousActions[4];
        bool isShoot = actionBuffers.DiscreteActions[0] == 1;
        movementInput = new Vector3(x, y, z);
        rotateInput = new Vector2(qx, qy);
        
        CustomFixedUpdate();
        if (isShoot)
        {
            OnShoot();
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
        // public override void Heuristic(float[] actionsOut)
    {
        var contActs = actionsOut.ContinuousActions;
        contActs[0] = movementInput.x;
        contActs[1] = movementInput.y;
        contActs[2] = movementInput.z;
        contActs[3] = rotateInput.x;
        contActs[4] = rotateInput.y;
        var discreteActs = actionsOut.DiscreteActions;
        discreteActs[0] = 0;
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(transform.rotation.eulerAngles.normalized); // 3
        sensor.AddObservation(rb.velocity); // 3
        sensor.AddObservation(rb.angularVelocity); // 3
        Dictionary<String, object> spaceObservations = spaceManager.GetObservations();
        sensor.AddObservation((Vector3)spaceObservations["gate"]); // 3
        List<Vector3> mineralsDists = (List<Vector3>)spaceObservations["mineralsDists"];
        List<Vector3> meteorsDists = (List<Vector3>)spaceObservations["meteorsDists"];
        sensor.AddObservation((float)spaceObservations["meteorsNumber"]); // 1
        for (int i = 0; i < 3; i++)
        {
            sensor.AddObservation(meteorsDists[i]);  // 9
        }
        sensor.AddObservation((float)spaceObservations["mineralsNumber"]); // 1
        for (int i = 0; i < 3; i++)
        {
            sensor.AddObservation(mineralsDists[i]); // 9
        }
        
        sensor.AddObservation(curShootCooldown); // 1
        // Total obs: 33
    }

    public override void OnEpisodeBegin()
    {
        transform.position = startPosition;
        transform.rotation = startRotate;
        rb.angularVelocity = Vector3.zero;
        rb.velocity = Vector3.zero;
        movementInput = Vector3.zero;
        rotateInput = Vector2.zero;
        curMinerals = 0.0f;
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
                range = Vector3.Distance(aimPosition.transform.position, hit.point);
                ParticleSystem hitEffect = Instantiate(shootEffect, hit.point, Quaternion.identity);
                Destroy(hitEffect.gameObject, 0.5f);
                if (hit.collider.CompareTag("Meteor"))
                {
                    MeteorScript meteor = hit.collider.GetComponent<MeteorScript>();
                    meteor.TakeHit();
                    AddReward(0.1f);
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
        AddReward(0.1f);
        
    }
    
    public void SellMinerals()
    {
        AddReward(curMinerals / maxMinerals * 10);
        curMinerals = 0.0f;
        cargoUI.SetCargo(0.0f);
    }
    
}